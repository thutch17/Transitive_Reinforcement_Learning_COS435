from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from utils.flax_utils import nonpytree_field

class TRLDataset:
    
    """ 
    sample dataloader
    """

    def __init__(self, dataset):
        """
        pseudocode:
        1. store observations and actions arrays
        2. find trajectory boundaries from terminals
           - ends = where terminals == 1
           - starts = [0] + (ends[:-1] + 1)
        3. store starts and lengths
        """
        self.dataset = dataset
        self.observations = dataset['observations']
        self.actions = dataset['actions']
        self.terminals = dataset['terminals']

        (self.ends,) = np.nonzero(self.terminals == 1)
        self.starts = np.concatenate([[0], self.ends[:-1] + 1])
        self.lengths = self.ends - self.starts + 1

    def sample(self, batch_size, rng):
        """
        sample (i, j, k) triples for the TRL value loss

        pseudocode:
        1. pick random trajectories
        2. sample i, then j > i, then k in [i, j-1]
        3. index into observations/actions arrays
        4. return dict:
           - s_i, a_i, s_j, a_j, s_k, a_k
           - leg1_len (k - i), leg2_len (j - k)
        """
        # step 1
        traj_rng, i_rng, j_rng, k_rng = jax.random.split(rng, 4)
        traj_idxs = np.asarray(jax.random.randint(traj_rng, (batch_size,), 0, len(self.starts)))
        starts = self.starts[traj_idxs]
        lengths = self.lengths[traj_idxs]

        # step 2
        max_i_offsets = np.maximum(lengths - 1, 1)
        i_offsets = np.asarray(jax.random.uniform(i_rng, (batch_size,)) * max_i_offsets).astype(int)
        i_idxs = starts + i_offsets

        # step 2
        remaining = lengths - i_offsets - 1
        j_span = np.maximum(remaining, 1)
        j_offsets = i_offsets + 1 + np.asarray(jax.random.uniform(j_rng, (batch_size,)) * j_span).astype(int)
        j_idxs = starts + j_offsets
        
        # step 2
        k_span = j_offsets - i_offsets
        k_offsets = i_offsets + np.asarray(jax.random.uniform(k_rng, (batch_size,)) * k_span).astype(int)
        k_idxs = starts + k_offsets

        # steps 3/4
        return {
            's_i': self.observations[i_idxs],
            'a_i': self.actions[i_idxs],
            's_j': self.observations[j_idxs],
            'a_j': self.actions[j_idxs],
            's_k': self.observations[k_idxs],
            'a_k': self.actions[k_idxs],
            'leg1_len': k_idxs - i_idxs,
            'leg2_len': j_idxs - k_idxs,
        }


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
        weight = 0.0
        if pred > target:
            weight = 1.0 - self.config['expectile']
        else:
            weight = self.config['expectile']
        return weight * base_loss

    def sample_behavioral_subgoals(self, batch):
        """pick midpoint states only from the same trajectory.

        paper reference:
        - section 4.2 says TRL only considers in-trajectory states as subgoals
        - need for stability random subgoals don't work
        - algorithm 1 samples i <= k <= j from the same trajectory chunk

        pseudocode:
        1. read a trajectory chunk (s_i, a_i, ..., s_j) from the batch
        2. sample an index k such that i <= k <= j
        3. set the subgoal state to s_k
        4. set the subgoal action to a_k when the q-based update needs it
        5. also keep track of the chunk lengths:
           - first leg length: k - i
           - second leg length: j - k
        6. return all of that

        ** note: might need to add something to datasets.py since idk if can do this
        type of sampling w normie OGBench
        """
        raise NotImplementedError('pseudocode only')

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
        6. return the target
        """
        raise NotImplementedError('pseudocode only')

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
        raise NotImplementedError('pseudocode only')

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
        raise NotImplementedError('pseudocode only')

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
        raise NotImplementedError('pseudocode only')


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
            encoder=ml_collections.config_dict.placeholder(str),
            policy_extraction='ddpgbc',
            use_oracle_distillation=False,
            # dataset stuff
            dataset_class='GCDataset',
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
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
