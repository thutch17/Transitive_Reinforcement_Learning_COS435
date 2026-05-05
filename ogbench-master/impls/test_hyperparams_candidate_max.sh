# Smoke test for Extension 3: value-guided in-trajectory candidate subgoal selection.
# This is only meant to check that TRL runs without config/shape errors.

python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/trl.py --agent.subgoal_strategy=candidate_max --agent.subgoal_num_candidates=4 --run_group=Test_Candidate_Max_Subgoals --train_steps=500 --eval_interval=500 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0.7 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=10
