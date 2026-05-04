# Hyperparameter testing for the different agents. This is not meant to be an exhaustive search, just a sanity check that the code runs with different hyperparameters.

# note for modifying from the original file: 
# offline_steps to train_steps,
# --agent.dataset to --agent,
# lam to distance_weight_lambda,
# pe_type to policy_extraction
# rpg to ddpgbc

# pointmaze-medium-navigate-v0 (TRL)
python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/trl.py --run_group=Test --train_steps=500 --eval_interval=500 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0.7 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=10

# pointmaze-medium-navigate-v0 (GCBC)
python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/gcbc.py --run_group=Test --train_steps=500 --eval_interval=500
# pointmaze-medium-navigate-v0 (GCIVL)
python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/gcivl.py --run_group=Test --train_steps=500 --eval_interval=500 --agent.alpha=10.0
# pointmaze-medium-navigate-v0 (GCIQL)
python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/gciql.py --run_group=Test --train_steps=500 --eval_interval=500 --agent.alpha=0.003
# pointmaze-medium-navigate-v0 (QRL)
python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/qrl.py --run_group=Test --train_steps=500 --eval_interval=500 --agent.alpha=0.0003
# pointmaze-medium-navigate-v0 (CRL)
python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/crl.py --run_group=Test --train_steps=500 --eval_interval=500 --agent.alpha=0.03
# pointmaze-medium-navigate-v0 (HIQL)
python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/hiql.py --run_group=Test --train_steps=500 --eval_interval=500 --agent.high_alpha=3.0 --agent.low_alpha=3.0