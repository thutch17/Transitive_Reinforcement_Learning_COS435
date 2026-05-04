# FROM THE ORIGINAL PAPER:
# humanoidmaze-giant-navigate-oraclerep-v0 (GCBC)
python main.py --env_name=humanoidmaze-giant-navigate-oraclerep-v0 --agent=agents/gcbc.py
# humanoidmaze-giant-navigate-oraclerep-v0 (GCIVL)
python main.py --env_name=humanoidmaze-giant-navigate-oraclerep-v0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.discount=0.995
# humanoidmaze-giant-navigate-oraclerep-v0 (GCIQL)
python main.py --env_name=humanoidmaze-giant-navigate-oraclerep-v0 --agent=agents/gciql.py --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-giant-navigate-oraclerep-v0 (QRL)
python main.py --env_name=humanoidmaze-giant-navigate-oraclerep-v0 --agent=agents/qrl.py --agent.alpha=0.001 --agent.discount=0.995
# humanoidmaze-giant-navigate-oraclerep-v0 (CRL)
python main.py --env_name=humanoidmaze-giant-navigate-oraclerep-v0 --agent=agents/crl.py --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-giant-navigate-oraclerep-v0 (HIQL)
python main.py --env_name=humanoidmaze-giant-navigate-oraclerep-v0 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
# humanoidmaze-giant-navigate-oraclerep-v0 (TRL)
python main.py --env_name=humanoidmaze-giant-navigate-oraclerep-v0 --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" --agent.discount=0.999 --dataset_replace_interval=4000 --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=0.1


# puzzle-4x5-play-oraclerep-v0 (GCBC)
python main.py --env_name=puzzle-4x5-play-oraclerep-v0 --agent=agents/gcbc.py
# puzzle-4x5-play-oraclerep-v0 (GCIVL)
python main.py --env_name=puzzle-4x5-play-oraclerep-v0 --agent=agents/gcivl.py --agent.alpha=10.0
# puzzle-4x5-play-oraclerep-v0 (GCIQL)
python main.py --env_name=puzzle-4x5-play-oraclerep-v0 --agent=agents/gciql.py --agent.alpha=1.0
# puzzle-4x5-play-oraclerep-v0 (QRL)
python main.py --env_name=puzzle-4x5-play-oraclerep-v0 --agent=agents/qrl.py --agent.alpha=0.3
# puzzle-4x5-play-oraclerep-v0 (CRL)
python main.py --env_name=puzzle-4x5-play-oraclerep-v0 --agent=agents/crl.py --agent.alpha=3.0
# puzzle-4x5-play-oraclerep-v0 (HIQL)
python main.py --env_name=puzzle-4x5-play-oraclerep-v0 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10
# puzzle-4x5-play-oraclerep-v0 (TRL)
python main.py --env_name=puzzle-4x5-play-oraclerep-v0 --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" --agent.discount=0.999 --dataset_replace_interval=1000 --agent.actor_geom_sample=True --agent.actor_p_trajgoal=0.5 --agent.actor_p_randomgoal=0.5 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=rejection



# puzzle-4x6-play-oraclerep-v0 (GCBC)
python main.py --env_name=puzzle-4x6-play-oraclerep-v0 --agent=agents/gcbc.py
# puzzle-4x6-play-oraclerep-v0 (GCIVL)
python main.py --env_name=puzzle-4x6-play-oraclerep-v0 --agent=agents/gcivl.py --agent.alpha=10.0
# puzzle-4x6-play-oraclerep-v0 (GCIQL)
python main.py --env_name=puzzle-4x6-play-oraclerep-v0 --agent=agents/gciql.py --agent.alpha=1.0
# puzzle-4x6-play-oraclerep-v0 (QRL)
python main.py --env_name=puzzle-4x6-play-oraclerep-v0 --agent=agents/qrl.py --agent.alpha=0.3
# puzzle-4x6-play-oraclerep-v0 (CRL)
python main.py --env_name=puzzle-4x6-play-oraclerep-v0 --agent=agents/crl.py --agent.alpha=3.0
# puzzle-4x6-play-oraclerep-v0 (HIQL)
python main.py --env_name=puzzle-4x6-play-oraclerep-v0 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10
# puzzle-4x6-play-oraclerep-v0 (TRL)
python main.py --env_name=puzzle-4x6-play-oraclerep-v0 --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" --agent.discount=0.999 --dataset_replace_interval=1000 --agent.actor_geom_sample=True --agent.actor_p_trajgoal=0.5 --agent.actor_p_randomgoal=0.5 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=rejection


# NEW TASKS:

# pointmaze-teleport-navigate-oraclerep-v0 (GCBC)
python main.py --env_name=pointmaze-teleport-navigate-oraclerep-v0 --agent=agents/gcbc.py
# pointmaze-teleport-navigate-oraclerep-v0 (GCIVL)
python main.py --env_name=pointmaze-teleport-navigate-oraclerep-v0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.discount=0.995
# pointmaze-teleport-navigate-oraclerep-v0 (GCIQL)
python main.py --env_name=pointmaze-teleport-navigate-oraclerep-v0 --agent=agents/gciql.py --agent.alpha=0.1 --agent.discount=0.995
# pointmaze-teleport-navigate-oraclerep-v0 (QRL)
python main.py --env_name=pointmaze-teleport-navigate-oraclerep-v0 --agent=agents/qrl.py --agent.alpha=0.001 --agent.discount=0.995
# pointmaze-teleport-navigate-oraclerep-v0 (CRL)
python main.py --env_name=pointmaze-teleport-navigate-oraclerep-v0 --agent=agents/crl.py --agent.alpha=0.1 --agent.discount=0.995
# pointmaze-teleport-navigate-oraclerep-v0 (HIQL)
python main.py --env_name=pointmaze-teleport-navigate-oraclerep-v0 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
# pointmaze-teleport-navigate-oraclerep-v0 (TRL)
python main.py --env_name=pointmaze-teleport-navigate-oraclerep-v0 --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" --agent.discount=0.999 --dataset_replace_interval=4000 --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=0.1

# antmaze-large-stitch-oraclerep-v0 (GCBC)
python main.py --env_name=antmaze-large-stitch-oraclerep-v0 --agent=agents/gcbc.py
# antmaze-large-stitch-oraclerep-v0 (GCIVL)
python main.py --env_name=antmaze-large-stitch-oraclerep-v0 --agent=agents/gcivl.py --agent.alpha=10.0
# antmaze-large-stitch-oraclerep-v0 (GCIQL)
python main.py --env_name=antmaze-large-stitch-oraclerep-v0 --agent=agents/gciql.py --agent.alpha=0.3
# antmaze-large-stitch-oraclerep-v0 (QRL)
python main.py --env_name=antmaze-large-stitch-oraclerep-v0 --agent=agents/qrl.py --agent.alpha=0.003
# antmaze-large-stitch-oraclerep-v0 (CRL)
python main.py --env_name=antmaze-large-stitch-oraclerep-v0 --agent=agents/crl.py --agent.alpha=0.1
# antmaze-large-stitch-oraclerep-v0 (HIQL)
python main.py --env_name=antmaze-large-stitch-oraclerep-v0 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0
# antmaze-large-stitch-oraclerep-v0 (TRL)
python main.py --env_name=antmaze-large-stitch-oraclerep-v0 --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0.0 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=0.7
