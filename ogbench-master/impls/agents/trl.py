from typing import Any

import copy
import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCDiscreteCritic, GCValue

class TRLAgent(flax.struct.PyTreeNode):
    """
    this is all pseudocode for now will get it fr this weekend but didn't want
    to be annoying and just start cranking code before anyone understands what's going
    on

    main sources to best understand high-level
    - section 4.2 explains how our in-trajectory subgoals are used
    - section 4.3.1 defines the TRL value loss in equation (11)
    - section 4.3.2 says policy extraction is done with either reparameterized gradients
      or rejection sampling
    - algorithm 1 gives the high-level training loop
    """

    # matching other agents
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def expectile_loss(self, pred, target, base_loss):
        """
        where this exists in the paper:
        - section 4.2 replaces the strict max over all subgoals with soft expectile regression
        - equation (10) is the conceptual expectile version of the transitive update
        - equation (11) uses the expectile variant d_kappa for the final TRL loss

        pseudocode:
        1. compare the current prediction to the target
        2. if pred is above target, use weight (1 - kappa)
        3. otherwise use weight kappa
        4. multiply the pointwise base loss by that weight
        5. return the weighted loss
        """
        # kind of a one liner so can pull into other if necessary just thought good to have func for each loss type
        weight = jnp.where(pred > target, 1.0 - self.config['expectile'], self.config['expectile'])
        return weight * base_loss

    def transitive_target(self, batch):
        """build the TRL target from two trajectory segments.

        paper reference:
        - equation (9) is the Q-based Bellman update we want here
        - section 4.3.1 turns this into the TRL value loss in equation (11)
        - the target is q_bar(s_i, a_i, s_k) * q_bar(s_k, a_k, s_j) (or edge cases)

        pseudocode:
        1. get subgoal batch from sample_behavioral_subgoals
        2. for the first leg:
           - if k-i <= 1, use gamma^(k-i)
           - else evaluate the target critic on (s_i, a_i, s_k)
        3. for the second leg:
           - if j-k <= 1, use gamma^(j-k)
           - else evaluate the target critic on (s_k, a_k, s_j)
        4. multiply the two leg values together
        5. return the target
        """
        # step 1
        subgoal_batch = batch

        # see other agents, take more conservative estimate
        def reduce_critic_output(values):
            if values.ndim > subgoal_batch['leg1_len'].ndim:
                return jnp.min(values, axis=0)
            return values

        # step 2
        discount = self.config['discount']
        first_leg_bootstrap = self.network.select('target_critic')(
            subgoal_batch['s_i'], subgoal_batch['s_k'], subgoal_batch['a_i']
        )
        first_leg_bootstrap = reduce_critic_output(first_leg_bootstrap)
        # edge condition k - i <= 1
        first_leg = jnp.where(
            subgoal_batch['leg1_len'] <= 1,
            discount ** subgoal_batch['leg1_len'],
            first_leg_bootstrap,
        )

        # step 3
        second_leg_bootstrap = self.network.select('target_critic')(
            subgoal_batch['s_k'], subgoal_batch['s_j'], subgoal_batch['a_k']
        )
        second_leg_bootstrap = reduce_critic_output(second_leg_bootstrap)
        # edge condition j-k <= 1
        second_leg = jnp.where(
            subgoal_batch['leg2_len'] <= 1,
            discount ** subgoal_batch['leg2_len'],
            second_leg_bootstrap,
        )

        # step 4/5
        return first_leg * second_leg

    def distance_weight(self, critic_prediction):
        """reweight samples so short chunks matter more b/c get bad cumluating bias if don't

        paper reference:
        - section 4.3.1 introduces distance-based re-weighting
        - longer chunks depend on shorter chunks being accurate first
        - equation (11) uses the weight w(s_i, s_j)
        - the weight is roughly inverse in estimated distance, controlled by lambda

        pseudocode:
        1. convert the critic prediction into an estimated distance
        2. compute w = 1 / (1 + estimated_distance)^lambda
        3. if lambda == 0, just return 1 for every sample
        4. return the weights
        """
        # step 3
        lam = self.config['distance_weight_lambda']
        if lam == 0.0:
            return jnp.ones_like(critic_prediction)
        # step 1 (ok so little funky but from looking it seems you want to log transform in most cases?)
        # like supposedly we want super long horizon to shrink weight exponentially but can change this
        estimated_distance = -jnp.log(jnp.clip(critic_prediction, a_min=1e-8, a_max=1.0))
        # step 2
        w = 1.0 / (1.0 + estimated_distance) ** lam
        return w

    def critic_loss(self, batch, grad_params):
        """main TRL critic objective.

        paper reference:
        - called value loss in paper, just mathcing the other agents
        - section 4.3.1 gives the reasoning
        - equation (11) defines the TRL value loss
        - they use binary cross-entropy as the base loss in experiments
        - section 4.2 explains why expetcile regression is used instead of a hard max

        pseudocode:
        1. evaluate the online critic on q(s_i, a_i, s_j)
        2. build the transitive target with transitive_target()
        3. compute pointwise BCE between the critic prediction and the target
        4. wrap that with expectile weighting
        5. compute the distance-based sample weights with distance_weight
        6. multiply the loss by those weights
        7. average across the batch and return

        """
        # step 1
        # probably debug this later not sure if we should be outputting logits but i think so with BCE?
        critic_logits = self.network.select('critic')(batch['s_i'], batch['s_j'], batch['a_i'], params=grad_params)
        if critic_logits.ndim > batch['leg1_len'].ndim:
            critic_logits = jnp.min(critic_logits, axis=0)
        critic_prediction = jax.nn.sigmoid(critic_logits)

        # step 2
        target = self.transitive_target(batch)

        # step 3
        base_loss = optax.sigmoid_binary_cross_entropy(critic_logits, target)

        # step 4
        expectile_loss = self.expectile_loss(critic_prediction, target, base_loss)

        # step 5
        sample_weights = self.distance_weight(critic_prediction)

        # steps 6/7
        critic_loss = (expectile_loss * sample_weights).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'pred_mean': critic_prediction.mean(),
            'pred_max': critic_prediction.max(),
            'pred_min': critic_prediction.min(),
            'target_mean': target.mean(),
            'target_max': target.max(),
            'target_min': target.min(),
            'weight_mean': sample_weights.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """policy extraction objective.

        where this exists in the paper:
        - again named to match agents, how policy is extracted
        - section 4.3.2 says TRL first learns the value, then extracts a policy
        - paper uses reparameterized gradients in equation 4 (not super clear)

        pseudocode for the default path:
        1. sample or reparameterize an action a_pi ~ pi(. | s, g)
        2. evaluate Q(s, a_pi, g) with the learned critic
        3. compute the DDPG+BC style objective (equation 4):
           q(s, a_pi, g) + alpha * log pi(a | s, g)
        4. maximize that objective, or equivalently minimize its negative
        5. return the actor loss and logging stats
        """
        # step 1
        dist = self.network.select('actor')(
            batch['s_i'], batch['s_j'], params=grad_params
        )
        rng = rng if rng is not None else self.rng
        actions = dist.sample(seed=rng)

        # step 2
        q_vals = self.network.select('critic')(
            batch['s_i'], batch['s_j'], actions, params=grad_params
        )
        # again take min? this can change but want conservative updates
        if q_vals.ndim > batch['leg1_len'].ndim:
            q_vals = jnp.min(q_vals, axis=0)

        # step 3/4: log prob
        log_prob = dist.log_prob(actions)
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

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """combine every loss term that should be optimized together.

        paper reference:
        - algorithm 1 separates value learning and policy extraction, but also
          these can be run in parallel (yay jax)
        - ogbench agents expose one 'total_loss' entry point for optimization

        pseudocode:
        1. get actor and critic losses from functions
        2. do appropriate dimensioning and summing and return
        """
        info = {}
        rng = rng if rng is not None else self.rng

        #step 1
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            # the slash is arbitrary but use when indexing (ie critic/loss)
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        # step 2
        loss = self.config['critic_loss_weight'] * critic_loss + self.config['actor_loss_weight'] * actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """update the target critic.

        paper reference:
        - section 4.2 and equation (10) / equation (11) use a target network q_bar
        - so the code needs a target critic and a way to keep it synced but not overcorect
        - ogbench agents do this with "polyak" averaging (thanks Google)
        - this is not like a paper thing really just open ended target network

        pseudocode:
        1. read the online parameters
        2. read the target parameters
        3. blend them with tau
        4. write the new target params
        """

        def polyak_average(p, tp):
            return p * self.config['tau'] + tp * (1 - self.config['tau'])

        # ok i'm probably messing up this jax/parameters thing since had to be googling
        new_target_params = jax.tree_util.tree_map(
            # step 3
            polyak_average,
            # step 1 (the current critic weights)
            self.network.params[f'modules_{module_name}'],
            # step 2 (lagged copy for stability)
            self.network.params[f'modules_target_{module_name}'],
        )
        # step 4
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """one training step.

        paper reference:
        - algorithm 1 is an iterative training loop over dataset trajectory chunks
        - in ogbench, each agent exposes an 'update' function that does exactly one step

        pseudocode:
        1. split rng
        2. run one optimizer step on the network params
        3. update the target critic
        4. return the new agent & log stuff
        """
        # step 1
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        # step 2
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        # step 3
        self.target_update(new_network, 'critic')

        # step 4
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """action selection for rollouts / evaluation.

        paper reference:
        - after policy extraction, the learned policy has to be queried when testing/evaluating
        - rolling with direct actor as opposed to rejection-sampling (see 4.3.2, equations 4/5)

        pseudocode:
        1. evaluate pi(a | s, g)
        2. sample or take the mode
        3. clip continuous actions if needed
        """
        # step 1
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        # step 2
        actions = dist.sample(seed=seed)
        # step 3
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """construct the TRL agent and its modules.

        paper reference:
        - another key function for the ogbench agents
        - section 4.3 says TRL learns a goal-conditioned Q and then extracts a policy
        - equation (11) needs an online critic and a target critic
        - section 4.3.2 needs a policy module for extraction

        pseudocode:
        1. initialize the rng
        2. get the action and other dimensions dimension
        3. build optional encoders if the environment is visual
        4. build:
           - critic: Q(s, a, g)
           - target_critic: slowly updated copy of Q
           - actor: pi(a | s, g)
        5. initialize params from example inputs
        6. create the optimizer state
        7. return the agent
        """
        # step 1
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        # step 2
        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # step 3
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        # step 4a (not positive if all required args here?)
        if config['discrete']:
            critic_def = GCDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                action_dim=action_dim,
            )
        else:
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
            )

        # step 4c
        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        # step 5
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            # step 4b (same definition)
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']

        # step 6
        network_tx = optax.adam(learning_rate=config['lr'])
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # step 7
        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """starter config for a trl implementation.

    extra fields:
    - `expectile` is needed because section 4.2 / equation (11) use expectile regression
    - `distance_weight_lambda` is needed because section 4.3.1 introduces re-weighting
    - `alpha` is needed for reparameterized policy extraction from equation (4)
    - `dataset_class` is left as `GCDataset` for now (need to figure this out)
    """
    config = ml_collections.ConfigDict(
        dict(
            # agent stuff
            agent_name='trl',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            alpha=0.1,
            expectile=0.7,
            distance_weight_lambda=0.0,
            critic_loss_weight=1.0,
            actor_loss_weight=1.0,
            const_std=True,
            discrete=False,
            encoder=None,
            policy_extraction='ddpgbc',
            use_oracle_distillation=False,
            # dataset stuff
            dataset_class='TRLDataset',
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=False,
            p_aug=0.0,
            frame_stack=None,
        )
    )
    return config
