# Run TRL midpoint-subgoal extension:
# Extension 1: Balanced In-Trajectory Subgoal Selection

# 3 long-horizon tasks from original paper:

# humanoidmaze-giant-navigate-v0 (TRL)
python main.py --env_name=humanoidmaze-giant-navigate-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" --agent.discount=0.999 --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=ddpgbc --agent.alpha=0.1

# puzzle-4x5-play-v0 (TRL)
python main.py --env_name=puzzle-4x5-play-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" --agent.discount=0.999 --agent.actor_geom_sample=True --agent.actor_p_trajgoal=0.5 --agent.actor_p_randomgoal=0.5 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=rejection

# python main.py --env_name=puzzle-4x5-play-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" --agent.discount=0.999 --agent.actor_geom_sample=True --agent.actor_p_trajgoal=0.5 --agent.actor_p_randomgoal=0.5 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=ddpgbc --agent.alpha=0.1

# puzzle-4x6-play-v0 (TRL)
python main.py --env_name=puzzle-4x6-play-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" --agent.discount=0.999 --agent.actor_geom_sample=True --agent.actor_p_trajgoal=0.5 --agent.actor_p_randomgoal=0.5 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=rejection

# python main.py --env_name=puzzle-4x6-play-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" --agent.discount=0.999 --agent.actor_geom_sample=True --agent.actor_p_trajgoal=0.5 --agent.actor_p_randomgoal=0.5 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=ddpgbc --agent.alpha=0.1


# NEW TASKS:

# pointmaze-teleport-navigate --v0 (TRL)
python main.py --env_name=pointmaze-teleport-navigate-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=0.7 --agent.policy_extraction=ddpgbc --agent.alpha=0.1

# antmaze-large-stitch --v0 (TRL)
python main.py --env_name=antmaze-large-stitch-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=0.0 --agent.policy_extraction=ddpgbc --agent.alpha=0.1

# additional tests:

python main.py --env_name=puzzle-3x3-play-oraclerep-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0.5 --agent.policy_extraction=ddpgbc --agent.alpha=0.1

python main.py --env_name=puzzle-4x4-play-oraclerep-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=2.0 --agent.policy_extraction=ddpgbc --agent.alpha=0.1

python main.py --env_name=humanoidmaze-medium-navigate-oraclerep-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.995 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0.0 --agent.policy_extraction=ddpgbc --agent.alpha=0.1

python main.py --env_name=humanoidmaze-large-navigate-oraclerep-v0 --agent=agents/trl.py --agent.subgoal_strategy=midpoint --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.995 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0.1 --agent.policy_extraction=ddpgbc --agent.alpha=0.1
