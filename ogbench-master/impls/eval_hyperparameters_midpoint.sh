# Run TRL midpoint-subgoal extension first:
# This is the same as eval_hyperparameters.sh, except we add:
#   --agent.subgoal_strategy=midpoint
# to each TRL command.
#
# Baseline TRL uses --agent.subgoal_strategy=uniform by default.
# Midpoint TRL keeps subgoals in-trajectory but chooses a balanced midpoint k
# instead of sampling k uniformly.

# 3 long-horizon tasks from original paper:

# humanoidmaze-giant-navigate-v0 (TRL midpoint)
python main.py --env_name=humanoidmaze-giant-navigate-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" --agent.discount=0.999 --dataset_replace_interval=4000 --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=ddpgbc --agent.alpha=0.1

# puzzle-4x5-play-v0 (TRL midpoint)
python main.py --env_name=puzzle-4x5-play-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" --agent.discount=0.999 --dataset_replace_interval=1000 --agent.actor_geom_sample=True --agent.actor_p_trajgoal=0.5 --agent.actor_p_randomgoal=0.5 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=rejection

# puzzle-4x6-play-v0 (TRL midpoint)
python main.py --env_name=puzzle-4x6-play-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" --agent.discount=0.999 --dataset_replace_interval=1000 --agent.actor_geom_sample=True --agent.actor_p_trajgoal=0.5 --agent.actor_p_randomgoal=0.5 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=rejection


# NEW TASKS:

# pointmaze-teleport-navigate-v0 (TRL midpoint)
python main.py --env_name=pointmaze-teleport-navigate-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=0.7 --agent.policy_extraction=ddpgbc --agent.alpha=0.1

# antmaze-large-stitch-v0 (TRL midpoint)
python main.py --env_name=antmaze-large-stitch-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=0.0 --agent.policy_extraction=ddpgbc --agent.alpha=0.1
