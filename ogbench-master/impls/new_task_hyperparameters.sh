### RUNNING BASELINES for the OG-Bench benchmark:

# Then run the other agents with the hyperparameters from the original paper.

# pointmaze-teleport-navigate-v0 (GCBC)
python main.py --env_name=pointmaze-teleport-navigate-v0 --agent=agents/gcbc.py
# pointmaze-teleport-navigate-v0 (GCIVL)
python main.py --env_name=pointmaze-teleport-navigate-v0 --agent=agents/gcivl.py --agent.alpha=10.0
# pointmaze-teleport-navigate-v0 (GCIQL)
python main.py --env_name=pointmaze-teleport-navigate-v0 --agent=agents/gciql.py --agent.alpha=0.003
# pointmaze-teleport-navigate-v0 (QRL)
python main.py --env_name=pointmaze-teleport-navigate-v0 --agent=agents/qrl.py --agent.alpha=0.0003
# pointmaze-teleport-navigate-v0 (CRL)
python main.py --env_name=pointmaze-teleport-navigate-v0 --agent=agents/crl.py --agent.alpha=0.03
# pointmaze-teleport-navigate-v0 (HIQL)
python main.py --env_name=pointmaze-teleport-navigate-v0 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0
# pointmaze-teleport-navigate --v0 (TRL)
python main.py --env_name=pointmaze-teleport-navigate-v0 --agent=agents/trl.py --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=False --agent.distance_weight_lambda=0 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=0.1


# antmaze-large-stitch-v0 (GCBC)
python main.py --env_name=antmaze-large-stitch-v0 --agent=agents/gcbc.py
# antmaze-large-stitch-v0 (GCIVL)
python main.py --env_name=antmaze-large-stitch-v0 --agent=agents/gcivl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=10.0
# antmaze-large-stitch-v0 (GCIQL)
python main.py --env_name=antmaze-large-stitch-v0 --agent=agents/gciql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3
# antmaze-large-stitch-v0 (QRL)
python main.py --env_name=antmaze-large-stitch-v0 --agent=agents/qrl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003
# antmaze-large-stitch-v0 (CRL)
python main.py --env_name=antmaze-large-stitch-v0 --agent=agents/crl.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1
# antmaze-large-stitch-v0 (HIQL)
python main.py --env_name=antmaze-large-stitch-v0 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.high_alpha=3.0 --agent.low_alpha=3.0
# antmaze-large-stitch --v0 (TRL)
python main.py --env_name=antmaze-large-stitch-v0 --agent=agents/trl.py --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=False --agent.distance_weight_lambda=0.0 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=0.7
