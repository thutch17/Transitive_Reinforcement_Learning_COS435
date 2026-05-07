# these are the hyperparameters for the evaluation notebook (Veronica's)
# it runs the non-oraclerep 3x3 puzzle for 500 steps

# pointmaze-medium-navigate-v0 (TRL)
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/trl.py --run_group=Test_3 --run_group=Test --train_steps=500 --eval_interval=500 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0.7 --agent.policy_extraction=ddpgbc --agent.alpha=2
