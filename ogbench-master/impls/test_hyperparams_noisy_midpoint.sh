# Hyperparameter testing / smoke test for TRL noisy midpoint subgoal strategy.
# This is not meant to measure final performance, just to sanity-check that the code runs.

# note for modifying from the original file: 
# offline_steps to train_steps,
# --agent.dataset to --agent,
# lam to distance_weight_lambda,
# pe_type to policy_extraction
# rpg to ddpgbc

# pointmaze-medium-navigate-v0 (TRL noisy midpoint)
python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/trl.py --agent.subgoal_strategy=noisy_midpoint --run_group=Test --train_steps=500 --eval_interval=500 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0.7 --agent.policy_extraction=ddpgbc --agent.alpha=0.1
