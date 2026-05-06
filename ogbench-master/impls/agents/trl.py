from typing import Any

import copy
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCDiscreteCritic, GCValue, ActorVectorField

class TRLAgent(flax.struct.PyTreeNode):
    """Divide and conquer value learning for GCRL

    main sources to best understand high-level
    - section 4.2 explains how our in-trajectory subgoals are used
    - section 4.3.1 defines the TRL value loss in equation (11)
    - section 4.3.2 says policy extraction is done with either reparameterized gradients
      or rejection sampling
    - algorithm 1 gives the high-level training loop
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def expectile_loss(self, critic_logits, target_labels):
        """Compute the expectile variant of binary cross-entropy loss.
        
        Section 3.1 defines the (conceptual) value loss used in Equation (10) as
        the expectation of the expectile loss:
            \ell_{\kappa}^{2}(x) = |\kappa - 1_{x < 0}| * x^2
        **This is equivalent to what was defined in expectile_loss() in gciql.py**

        In practice, Section 4.3.1 indicates that we optimize for the expectile
        variant used in Equation (11):
            D_{\kappa}(x, y) = |\kappa - 1_{x > y}| * D(x, y)
        """
        critic_labels = jax.nn.sigmoid(critic_logits)
        
        # 1. Compute the binary cross-entropy loss
        loss = optax.sigmoid_binary_cross_entropy(critic_logits, target_labels)
        weight = jnp.where(
            # 2. Evaluate indicator function
            critic_labels <= target_labels, # indicator function should compare probability labels
            # 3. If x <= y: weight = \kappa
            self.config['expectile'],
            # 4. If x > y:  weight = |\kappa - 1| for 0 < \kappa < 1
            1 - self.config['expectile']
        )

        # 5. Return the weighted loss
        return weight * loss

    def transitive_target(self, batch):
        """Compute the TRL target value:
            \bar{Q}(s_i, a_i, s_k) * \bar{Q}(s_k, a_k, s_j),
        from two trajectory segments.

        Equation (9) establishes the Q-based Bellman update. Section 4.3.1 uses
        this update rule in the TRL value loss computation in Equation (11).
        """
        
        # 1. Get subgoal batch
        subgoal_batch = batch

        def reduce_critic_output(values): # today I learned you can define a function inside a function in python, who knew
            # Identify critic in ensemble with most conservative target estimate
            if values.ndim > subgoal_batch['leg1_len'].ndim:
                return jnp.min(values, axis=0)
            return values

        discount = self.config['discount']
        
        # Select goal keys based on oracle distillation
        goal_key_k = 'g_k_obs' if self.config['use_oracle_distillation'] else 'g_k'
        goal_key_j = 'g_j_obs' if self.config['use_oracle_distillation'] else 'g_j'

        # 2. Compute the target of the first trajectory chunk
        #    If k - i <= 1: \bar{Q}(s_i, a_i, s_k) = \gamma^{k - i}
        first_leg_logits = self.network.select('target_critic')(
            subgoal_batch['s_i'], subgoal_batch[goal_key_k], subgoal_batch['a_i']
        )
        first_leg_logits = reduce_critic_output(first_leg_logits)

        first_leg_labels = jax.nn.sigmoid(first_leg_logits)

        first_leg_labels = jnp.where(
            subgoal_batch['leg1_len'] <= 1,
            discount ** subgoal_batch['leg1_len'],
            first_leg_labels,
        )

        # 3. Compute the target of the second trajectory chunk
        #    If j - k <= 1: \bar{Q}(s_k, a_k, s_j) = \gamma^{j - k}
        second_leg_logits = self.network.select('target_critic')(
            subgoal_batch['s_k'], subgoal_batch[goal_key_j], subgoal_batch['a_k']
        )

        second_leg_logits = reduce_critic_output(second_leg_logits)

        second_leg_labels = jax.nn.sigmoid(second_leg_logits)

        second_leg_labels = jnp.where(
            subgoal_batch['leg2_len'] <= 1,
            discount ** subgoal_batch['leg2_len'],
            second_leg_labels,
        )

        # 4. Return the product of the target logits of the two trajectory chunks
        return first_leg_labels * second_leg_labels

    def distance_weight(self, critic_logits):
        """Reweight samples so short chunks matter more b/c get bad cumluating bias if don't

        The accuracy of the target value for a longer trajectory chunk (s_i to
        s_j) depends on the accuracy of the target values for the two shorter
        trajectory chunks (s_i to s_k and s_k to s_j). Section 4.3.1 proposes
        distance-based re-weighting, in which the loss for each sample (s_i, s_j)
        is weighted by the factor:
            w(s_i, s_j) = (1 + \log_{\gamma} Q(s_i, a_i, s_j))^{-\lambda}
        
        The resulting weight for each trajectory chunk is (roughly) inversely
        proportional to its estimated distance, yielding a higher weight to
        shorter trajectory chunks.

        pseudocode:
        1. convert the critic prediction into an estimated distance
        """
        lam = self.config['distance_weight_lambda']
        critic_labels = jax.nn.sigmoid(critic_logits)
        
        # 1. If \lambda = 0: return 1 for every sample
        if lam == 0.0:
            return jnp.ones_like(critic_labels)
        
        # 1.5. The clipping isn't stricly neccesary but would help with numerical stability
        critic_labels_clipped = jnp.clip(critic_labels, a_min=1e-8, a_max=1.0 - 1e-8)

        # 2. Compute \log_{\gamma} Q(s_i, a_i, s_j)
        estimated_distance = jnp.log(critic_labels_clipped) / jnp.log(self.config['discount']) #this looks fine to me...
        
        # 3. Compute distance-based re-weights
        weights = 1.0 / ((1.0 + estimated_distance) ** lam)

        # 4. Return distance-based re-weights
        return weights

    def critic_loss(self, batch, grad_params):
        """Compute the TRL critic/value loss.

        Section 4.3.1 defines the value loss as (Equation 11):
            L^{TRL}(Q) = E_{\tau \sim D}[
                w(s_i, s_j) * D_{\kappa}(
                    Q(s_i, a_i, s_j), \bar{Q}(s_i, a_i, s_k) * \bar{Q}(s_k, a_k, s_j)
                )
            ]
        where D is the expectile variant of the binary cross-entropy loss.
        """
        # NOTE (regarding logits vs. labels):     
        # After doing some searching, you need to feed in logits to the expectile loss
        # for the sigmoid binary cross-entropy loss to be well-behaved
        # but for the distance based reweighing you need to convert to labels
        # so that inputs are in (0, 1) and the loss is well-behaved. 
        # In general, if something looks like it should be a probability or bounded between 0 and 1, it probably is a label
        # For consistancy's sake I labeled what goes into the expectile loss as "logits" and what goes into the distance weight as "labels"

        # 1. Evaluate the student critic on Q(s_i, a_i, g_j)
        # Use oracle goal observations when distilling, learned goals otherwise
        goal_key = 'g_j_obs' if self.config['use_oracle_distillation'] else 'g_j'
        critic_logits = self.network.select('critic')(
            batch['s_i'], batch[goal_key], batch['a_i'], params=grad_params
        )
        if critic_logits.ndim > batch['leg1_len'].ndim:
            critic_logits = jnp.min(critic_logits, axis=0)

        # 2. Evaluate the target critic on \bar{Q}(s_i, a_i, s_k) * \bar{Q}(s_k, a_k, s_j)
        target_labels = self.transitive_target(batch)

        # 3. Compute the expectile variant of the binary cross-entropy loss between
        #    the online critic prediction and target critic
        expectile_loss = self.expectile_loss(critic_logits=critic_logits, target_labels=target_labels)

        # 4. Compute distance-based re-weighted loss
        weights = self.distance_weight(critic_logits=critic_logits)
        critic_loss = (expectile_loss * weights).mean()

        # is using oracle distillation, also compute distillation loss and add to critic loss
        if self.config['use_oracle_distillation']:
            oracle_logits = self.network.select('oracle_critic')(
                batch['s_i'], batch['g_j'], batch['a_i'], params=grad_params
            )

            oracle_distill_loss = optax.sigmoid_binary_cross_entropy(
                oracle_logits, jax.lax.stop_gradient(jax.nn.sigmoid(critic_logits))
            ).mean()

            critic_loss = critic_loss + oracle_distill_loss

        return critic_loss, {
            'critic_loss': critic_loss,
            'pred_mean': critic_logits.mean(),
            'pred_max': critic_logits.max(),
            'pred_min': critic_logits.min(),
            'target_mean': target_labels.mean(),
            'target_max': target_labels.max(),
            'target_min': target_labels.min(),
            'weight_mean': weights.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """policy extraction objective.

        where this exists in the paper:
        - again named to match agents, how policy is extracted
        - section 4.3.2 says TRL first learns the value, then extracts a policy
        - paper uses reparameterized gradients in equation 4 (not super clear)

        pseudocode for the default path (reparameterized gradients aka ddpg+bc):
        1. sample or reparameterize an action a_pi ~ pi(. | s, g)
        2. evaluate Q(s, a_pi, g) with the learned critic
        3. compute the DDPG+BC style objective (equation 4):
           q(s, a_pi, g) + alpha * log pi(a | s, g)
        4. maximize that objective, or equivalently minimize its negative
        5. return the actor loss and logging stats

        This should branch for the rejection sampling policy extraction method, but that is not currently implemented.
        """
        if self.config['policy_extraction'] == 'ddpgbc':
            # step 1
            dist = self.network.select('actor')(
                batch['s_i'], batch['g_j'], params=grad_params
            )
            rng = rng if rng is not None else self.rng
            actions = dist.sample(seed=rng)

            # step 2
            critic_module = 'oracle_critic' if self.config['use_oracle_distillation'] else 'critic'
            q_vals = self.network.select(critic_module)(
                batch['s_i'], batch['g_j'], actions
            )
            # again take min? this can change but want conservative updates
            if q_vals.ndim > batch['leg1_len'].ndim:
                q_vals = jnp.min(q_vals, axis=0)

            # step 3/4: log prob
            log_prob = dist.log_prob(batch['a_i']) # should be computing pi(a | s, g), not pi(a_pi | s, g)
            alpha = self.config['alpha']
            objective = q_vals + alpha * log_prob

            # step 5
            actor_loss = -objective.mean()
            return actor_loss, {
                'actor_loss': actor_loss,
                'q_mean': q_vals.mean(),
                'q_max': q_vals.max(),
                'q_min': q_vals.min(),
                'log_prob_mean': log_prob.mean(),
            }
        
        # gets the actor loss from rejection sampling with flow model
        elif self.config['policy_extraction'] == 'rejection':
            rng = rng if rng is not None else self.rng
            batch_size, action_dim = batch['a_i'].shape
            x_rng, t_rng = jax.random.split(rng, 2)

            x_0 = jax.random.normal(x_rng, shape=(batch_size, action_dim))
            x_1 = batch['a_i']
            t = jax.random.uniform(t_rng, shape=(batch_size, 1))
            x_t = t * x_1 + (1 - t) * x_0
            y = x_1 - x_0

            pred = self.network.select('actor')(batch['s_i'], batch['g_j'], x_t, t, params = grad_params)

            critic_module = 'oracle_critic' if self.config['use_oracle_distillation'] else 'critic'
            q_vals = self.network.select(critic_module)(batch['s_i'], batch['g_j'], batch['a_i'])
            if q_vals.ndim > batch['leg1_len'].ndim:
                q_vals = jnp.min(q_vals, axis=0)

            actor_loss = jnp.mean((pred - y) ** 2)
            return actor_loss, {
                "actor_loss": actor_loss,
                "pred_mean": pred.mean(),
                "pred_max": pred.max(),
                'q_mean': q_vals.mean(),
                'q_max': q_vals.max(),
                'q_min': q_vals.min(),
            }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss.

        paper reference:
        - algorithm 1 separates value learning and policy extraction, but also
          these can be run in parallel (yay jax)
        - ogbench agents expose one 'total_loss' entry point for optimization
        **Code should be the same from all other agent implementations (like gciql.py)!**
        """
        info = {}
        rng = rng if rng is not None else self.rng

        # 1. Compute actor and critic losses.
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            # the slash is arbitrary but use when indexing (ie critic/loss)
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        # 2. Compute total loss.
        loss = (
            self.config['critic_loss_weight'] * critic_loss + 
            self.config['actor_loss_weight'] * actor_loss
        )
        return loss, info

    def target_update(self, network, module_name):
        """update the target critic.

        paper reference:
        - section 4.2 and equation (10) / equation (11) use a target network q_bar
        - so the code needs a target critic and a way to keep it synced but not overcorect
        - ogbench agents do this with "polyak" averaging (thanks Google)
        - this is not like a paper thing really just open ended target network
        **Code be should the same from all other agent implementations (like gciql.py)!**
        """
        new_target_params = jax.tree_util.tree_map(
            # 3. Compute Polyak average of model parameters
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            # 1. Read current critic parameters.
            network.params[f'modules_{module_name}'],
            # 2. Read target critic parameters (lagged copy for stability).
            network.params[f'modules_target_{module_name}'],
        )

        # 4. Write new target parameters and return updated network.
        new_params = {
            k: new_target_params if k == f'modules_target_{module_name}' else v
            for k, v in network.params.items()
        }
        return network.replace(params=new_params)

    @jax.jit
    def update(self, batch):
        """Update the agent for one training step and return a new agent with
        information dictionary.

        Args:
            batch: dict of sample (i, j, k) triples
        
        paper reference:
        - algorithm 1 is an iterative training loop over dataset trajectory chunks
        - in ogbench, each agent exposes an 'update' function that does exactly one step
        """
        # 1. Initialize RNG subkey
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        # 2. Run one optimizer step on network params
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        # 3. Update target critic
        new_network = self.target_update(new_network, 'critic')

        # 4. Return new agent and information dictionary
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from actor network for rollouts / evaluation.

        Section 3.1 and 4.3.2 extracts a policy that maximizes the learned value
        either by:
         - Equation (4): Reparametrized Gradients - maximizing the DDPG+BC objective
         - Equation (5): Rejection Sampling - maximizing the value function
        By default, policy extraction is completed by reparametrized gradients.
        """
        # Extract policy according to reparametrized gradients
        if self.config['policy_extraction'] == 'ddpgbc':
            # 1. Compute \pi(a | s, g) \forall a \in A, where \pi is previously
            #    updated to maximize the DDPG+BC objective
            dist = self.network.select('actor')(observations, goals, temperature=temperature)
            # 2. Sample action
            actions = dist.sample(seed=seed)
            # 3. Clip continuous actions
            if not self.config['discrete']:
                actions = jnp.clip(actions, -1, 1)

        # Extract policy according to rejection sampling
        elif self.config['policy_extraction'] == 'rejection':
            # equation: \pi(a | s, g) argmax_{a_1, ... a_n, a_i ~ \pi^\beta (a | s, g)} Q(s, a_i, g)
            # where \pi^\beta is a goal-condition BC policy separate from the actual policy
            # the BC policy is modeled by an expressive generation model (diffusion mode) and flow matching

            pe_info = self.config['rejection']

            n_obs = jnp.repeat(jnp.expand_dims(observations, 0), repeats=pe_info.num_samples, axis=0) # 
            n_goals = jnp.repeat(jnp.expand_dims(goals, 0), repeats=pe_info.num_samples, axis=0) 
            n_actions = jax.random.normal(seed, (
                    pe_info.num_samples,
                    *observations.shape[:-1],
                    self.network.model_def.modules['actor'].action_dim,
                ),
            )

            # for each time step, update the sampled actions with the velocity from the flow model
            for i in range(pe_info.flow_steps):
                t = jnp.full(
                    (pe_info.num_samples, *observations.shape[:-1], 1),
                    i / pe_info.flow_steps,
                )
                vels = self.network.select('actor')(n_obs, n_goals, n_actions, t) # get velocity from flow model
                n_actions = n_actions + vels / pe_info.flow_steps # update actions with velocity
            
            n_actions = jnp.clip(n_actions, -1, 1) # clip actions to be in action space

            critic_module = 'oracle_critic' if self.config['use_oracle_distillation'] else 'critic'
            q = self.network.select(critic_module)(n_obs, goals=n_goals, actions=n_actions) # evaluate Q for all sampled actions
            if q.ndim > n_actions.ndim - 1:
                q = jnp.min(q, axis=0)

            # Determine a* that maximizes 
            if len(observations.shape) == 2:
                actions = n_actions[
                    jnp.argmax(q, axis=0), jnp.arange(observations.shape[0])
                ]
            else:
                actions = n_actions[jnp.argmax(q, axis=0)] # select action with highest Q for each observation
            
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        ex_goals=None,
    ):
        """Create a new agent.

        Args:
            seed: random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.

        paper reference:
        - another key function for the ogbench agents
        - section 4.3 says TRL learns a goal-conditioned Q and then extracts a policy
        - equation (11) needs an online critic and a target critic
        - section 4.3.2 needs a policy module for extraction
        """
        print("Update: TRLAgent.create is being called v1")
        
        # 1. Initialize the RNG key
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        # 2. Define action dimension from example observations
        # Use provided example goals when available; fall back to observations.
        ex_goals = ex_goals if ex_goals is not None else ex_observations
        ex_times = ex_actions[..., :1]
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # FLAGGED: TRL datasets are state-based, not pixel-based. Revisit if 
        #    we expand our set of environments to "visual-<ENV>"
        # encoders = dict()
        # if config['encoder'] is not None:
        #     encoder_module = encoder_modules[config['encoder']]
        #     encoders['critic'] = GCEncoder(concat_encoder=encoder_module())
        #     encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        # 3. Define actor-critic network
        if config['discrete']: # do we need this?
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                # final_fc_init_scale: float = 1e-2
                # gc_encoder=encoders.get('actor'),
            )
            critic_def = GCDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                # gc_encoder=encoders.get('critic'),
                action_dim=action_dim,
            )

            oracle_critic_def = GCDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                action_dim=action_dim,
            )

            ex_actor_input = (ex_observations, ex_goals)
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                # log_std_min=-5,
                # log_std_max=2,
                # tanh_squash: False,
                state_dependent_std=False,
                const_std=config['const_std'],
                # final_fc_init_scale: float = 1e-2
                # gc_encoder=encoders.get('actor'),
            )
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                # gc_encoder=encoders.get('critic'),
            )

            oracle_critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
            )

            ex_actor_input = (ex_observations, ex_goals)

        if config["policy_extraction"] == 'rejection':
            actor_def = ActorVectorField(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                layer_norm=config["layer_norm"],
            )
            ex_actor_input = (ex_observations, ex_goals, ex_actions, ex_times)

        # 4. Initialize network parameters
        # When using oracle distillation, the critic and oracle critic intentionally see different goal inputs.
        ex_critic_goals = ex_observations if config['use_oracle_distillation'] else ex_goals
        network_info = dict(
            actor=(actor_def, ex_actor_input),
            critic=(critic_def, (ex_observations, ex_critic_goals, ex_actions)),
            oracle_critic = (oracle_critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_critic_goals, ex_actions)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']
                
        # 5. Initialize critic and target critic with same parameters
        # unfreeze params if it's a FrozenDict to allow mutation (a bug fix from the original paper)
        network_params = dict(network_params)
        network_params['modules_target_critic'] = network_params['modules_critic']

        # 6. Initialize optimizers
        network_tx = optax.adam(learning_rate=config['lr'])
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # 7. Return the agent
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """
    - `alpha` is needed for reparameterized policy extraction from equation (4)
    """
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters
            agent_name='trl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.999,  # Discount factor.
            tau=0.005,  # Model parameter update rate for Polyak average.
            alpha=0.1,  # BC coefficient in DDPG+BC 
            expectile=0.7,  # Parameter for expectile regression loss.
            distance_weight_lambda=0.0,  # Distance-based re-weighting parameter. 
            critic_loss_weight=1.0,  # Coefficient for critic loss in total loss.
            actor_loss_weight=1.0,  # Coefficient for actor loss in total loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=None,  # Unused, all environments are state-based, not pixel-based.
            policy_extraction='rejection',  # Method ('ddpgbc' or 'rejection') for policy extraction.
            ddpgbc=ml_collections.ConfigDict(dict(alpha =0.03, const_std=True)), # hyperparameters for the ddpg+bc policy extraction method, used when policy_extraction='ddpgbc'
            rejection=ml_collections.ConfigDict(dict(num_samples=32, flow_steps=10)), # hyperparameters for the rejection sampling policy extraction method, used when policy_extraction='rejection'
            use_oracle_distillation=False, # Whether oracle distillation is used
            # Dataset hyperparameters.
            dataset_class='TRLDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.5,  # Probability of using a random state as the actor goal.
            actor_geom_sample=True,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=None,  # Number of frames to stack.
        )
    )
    return config
