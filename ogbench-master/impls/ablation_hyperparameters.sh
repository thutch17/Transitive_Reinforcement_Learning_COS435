# expectile kappa
# --agent.expectile=0.7
python main.py --env_name=humanoidmaze-giant-navigate-v0 --agent=agents/trl.py --train_steps=500 --eval_interval=500 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.995 --agent.expectile=0.7 --agent.distance_weight_lambda=0.0 --agent.policy_extraction=ddpgbc --agent.alpha=0.1
# --agent.expectile=0.5
python main.py --env_name=humanoidmaze-giant-navigate-v0 --agent=agents/trl.py --train_steps=500 --eval_interval=500 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.995 --agent.expectile=0.5 --agent.distance_weight_lambda=0.0 --agent.policy_extraction=ddpgbc --agent.alpha=0.1

# subgoal distributions
# --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 (in-traj)
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/trl.py --run_group=Test --train_steps=500 --eval_interval=500 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=0.7 --agent.policy_extraction=ddpgbc --agent.alpha=2
# --agent.actor_p_trajgoal=0.0 --agent.actor_p_randomgoal=1.0 (random)
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/trl.py --run_group=Test --train_steps=500 --eval_interval=500 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=0.0 --agent.actor_p_randomgoal=1.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=0.7 --agent.policy_extraction=ddpgbc --agent.alpha=2


# distance based lambda
# --agent.distance_weight_lambda=2
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/trl.py --run_group=Test --train_steps=500 --eval_interval=500 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=2 --agent.policy_extraction=ddpgbc --agent.alpha=2
# --agent.distance_weight_lambda=0
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/trl.py --run_group=Test --train_steps=500 --eval_interval=500 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=False --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=ddpgbc --agent.alpha=2