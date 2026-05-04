#this is the script used to run the final evaluation of all methods with the best hyperparameters. You can modify this script to run additional experiments with different hyperparameters if you wish.

# note for modifying from the original file: 
# offline_steps to train_steps,
# --agent.dataset to --agent,
# lam to distance_weight_lambda,
# pe_type to policy_extraction
# rpg to ddpgbc

# FROM THE ORIGINAL PAPER:

# scene-play-oraclerep-v0 (GCBC)
python main.py --env_name=scene-play-oraclerep-v0 --agent=agents/gcbc.py
# scene-play-oraclerep-v0 (GCIVL)
python main.py --env_name=scene-play-oraclerep-v0 --agent=agents/gcivl.py --agent.alpha=10.0
# scene-play-oraclerep-v0 (GCIQL)
python main.py --env_name=scene-play-oraclerep-v0 --agent=agents/gciql.py --agent.alpha=1.0
# scene-play-oraclerep-v0 (QRL)
python main.py --env_name=scene-play-oraclerep-v0 --agent=agents/qrl.py --agent.alpha=0.3
# scene-play-oraclerep-v0 (CRL)
python main.py --env_name=scene-play-oraclerep-v0 --agent=agents/crl.py --agent.alpha=3.0
# scene-play-oraclerep-v0 (HIQL)
python main.py --env_name=scene-play-oraclerep-v0 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10
# scene-play-oraclerep-v0 (TRL)
python main.py --env_name=scene-play-oraclerep-v0 --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=1.0 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=1



# cube-single-play-oraclerep-v0 (GCBC)
python main.py --env_name=cube-single-play-oraclerep-v0 --agent=agents/gcbc.py
# cube-single-play-oraclerep-v0 (GCIVL)
python main.py --env_name=cube-single-play-oraclerep-v0 --agent=agents/gcivl.py --agent.alpha=10.0
# cube-single-play-oraclerep-v0 (GCIQL)
python main.py --env_name=cube-single-play-oraclerep-v0 --agent=agents/gciql.py --agent.alpha=1.0
# cube-single-play-oraclerep-v0 (QRL)
python main.py --env_name=cube-single-play-oraclerep-v0 --agent=agents/qrl.py --agent.alpha=0.3
# cube-single-play-oraclerep-v0 (CRL)
python main.py --env_name=cube-single-play-oraclerep-v0 --agent=agents/crl.py --agent.alpha=3.0
# cube-single-play-oraclerep-v0 (HIQL)
python main.py --env_name=cube-single-play-oraclerep-v0 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10
# cube-single-play-oraclerep-v0 (TRL)
python main.py --env_name=cube-single-play-oraclerep-v0 --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0.7 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=1



# cube-double-play-oraclerep-v0 (GCBC)
python main.py --env_name=cube-double-play-oraclerep-v0 --agent=agents/gcbc.py
# cube-double-play-oraclerep-v0 (GCIVL)
python main.py --env_name=cube-double-play-oraclerep-v0 --agent=agents/gcivl.py --agent.alpha=10.0
# cube-double-play-oraclerep-v0 (GCIQL)
python main.py --env_name=cube-double-play-oraclerep-v0 --agent=agents/gciql.py --agent.alpha=1.0
# cube-double-play-oraclerep-v0 (QRL)
python main.py --env_name=cube-double-play-oraclerep-v0 --agent=agents/qrl.py --agent.alpha=0.3
# cube-double-play-oraclerep-v0 (CRL)
python main.py --env_name=cube-double-play-oraclerep-v0 --agent=agents/crl.py --agent.alpha=3.0
# cube-double-play-oraclerep-v0 (HIQL)
python main.py --env_name=cube-double-play-oraclerep-v0 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10
# cube-double-play-oraclerep-v0 (TRL)
python main.py --env_name=cube-double-play-oraclerep-v0 --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=1 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=10



# puzzle-3x3-play-oraclerep-v0 (GCBC)
python main.py --env_name=puzzle-3x3-play-oraclerep-v0 --agent=agents/gcbc.py
# puzzle-3x3-play-oraclerep-v0 (GCIVL)
python main.py --env_name=puzzle-3x3-play-oraclerep-v0 --agent=agents/gcivl.py --agent.alpha=10.0
# puzzle-3x3-play-oraclerep-v0 (GCIQL)
python main.py --env_name=puzzle-3x3-play-oraclerep-v0 --agent=agents/gciql.py --agent.alpha=1.0
# puzzle-3x3-play-oraclerep-v0 (QRL)
python main.py --env_name=puzzle-3x3-play-oraclerep-v0 --agent=agents/qrl.py --agent.alpha=0.3
# puzzle-3x3-play-oraclerep-v0 (CRL)
python main.py --env_name=puzzle-3x3-play-oraclerep-v0 --agent=agents/crl.py --agent.alpha=3.0
# puzzle-3x3-play-oraclerep-v0 (HIQL)
python main.py --env_name=puzzle-3x3-play-oraclerep-v0 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10
# puzzle-3x3-play-oraclerep-v0 (TRL)
python main.py --env_name=puzzle-3x3-play-oraclerep-v0 --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0.5 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=2




# puzzle-4x4-play-oraclerep-v0 (GCBC)
python main.py --env_name=puzzle-4x4-play-oraclerep-v0 --agent=agents/gcbc.py
# puzzle-4x4-play-oraclerep-v0 (GCIVL)
python main.py --env_name=puzzle-4x4-play-oraclerep-v0 --agent=agents/gcivl.py --agent.alpha=10.0
# puzzle-4x4-play-oraclerep-v0 (GCIQL)
python main.py --env_name=puzzle-4x4-play-oraclerep-v0 --agent=agents/gciql.py --agent.alpha=1.0
# puzzle-4x4-play-oraclerep-v0 (QRL)
python main.py --env_name=puzzle-4x4-play-oraclerep-v0 --agent=agents/qrl.py --agent.alpha=0.3
# puzzle-4x4-play-oraclerep-v0 (CRL)
python main.py --env_name=puzzle-4x4-play-oraclerep-v0 --agent=agents/crl.py --agent.alpha=3.0
# puzzle-4x4-play-oraclerep-v0 (HIQL)
python main.py --env_name=puzzle-4x4-play-oraclerep-v0 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10
# puzzle-4x4-play-oraclerep-v0 (TRL)
python main.py --env_name=puzzle-4x4-play-oraclerep-v0 --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=2.0 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=2



# antmaze-large-navigate-oraclerep-v0 (GCBC)
python main.py --env_name=antmaze-large-navigate-oraclerep-v0 --agent=agents/gcbc.py
# antmaze-large-navigate-oraclerep-v0 (GCIVL)
python main.py --env_name=antmaze-large-navigate-oraclerep-v0 --agent=agents/gcivl.py --agent.alpha=10.0
# antmaze-large-navigate-oraclerep-v0 (GCIQL)
python main.py --env_name=antmaze-large-navigate-oraclerep-v0 --agent=agents/gciql.py --agent.alpha=0.3
# antmaze-large-navigate-oraclerep-v0 (QRL)
python main.py --env_name=antmaze-large-navigate-oraclerep-v0 --agent=agents/qrl.py --agent.alpha=0.003
# antmaze-large-navigate-oraclerep-v0 (CRL)
python main.py --env_name=antmaze-large-navigate-oraclerep-v0 --agent=agents/crl.py --agent.alpha=0.1
# antmaze-large-navigate-oraclerep-v0 (HIQL)
python main.py --env_name=antmaze-large-navigate-oraclerep-v0 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0
# antmaze-large-navigate-oraclerep-v0 (TRL)
python main.py --env_name=antmaze-large-navigate-oraclerep-v0 --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0.0 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=0.7



# antsoccer-arena-navigate-oraclerep-v0 (GCBC)
python main.py --env_name=antsoccer-arena-navigate-oraclerep-v0 --agent=agents/gcbc.py
# antsoccer-arena-navigate-oraclerep-v0 (GCIVL)
python main.py --env_name=antsoccer-arena-navigate-oraclerep-v0 --agent=agents/gcivl.py --agent.alpha=10.0
# antsoccer-arena-navigate-oraclerep-v0 (GCIQL)
python main.py --env_name=antsoccer-arena-navigate-oraclerep-v0 --agent=agents/gciql.py --agent.alpha=0.1
# antsoccer-arena-navigate-oraclerep-v0 (QRL)
python main.py --env_name=antsoccer-arena-navigate-oraclerep-v0 --agent=agents/qrl.py --agent.alpha=0.003
# antsoccer-arena-navigate-oraclerep-v0 (CRL)
python main.py --env_name=antsoccer-arena-navigate-oraclerep-v0 --agent=agents/crl.py --agent.alpha=0.3
# antsoccer-arena-navigate-oraclerep-v0 (HIQL)
python main.py --env_name=antsoccer-arena-navigate-oraclerep-v0 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0
# antsoccer-arena-navigate-oraclerep-v0 (TRL)
python main.py --env_name=antsoccer-arena-navigate-oraclerep-v0 --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0.5 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=0.3



# humanoidmaze-medium-navigate-oraclerep-v0 (GCBC)
python main.py --env_name=humanoidmaze-medium-navigate-oraclerep-v0 --agent=agents/gcbc.py
# humanoidmaze-medium-navigate-oraclerep-v0 (GCIVL)
python main.py --env_name=humanoidmaze-medium-navigate-oraclerep-v0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.discount=0.995
# humanoidmaze-medium-navigate-oraclerep-v0 (GCIQL)
python main.py --env_name=humanoidmaze-medium-navigate-oraclerep-v0 --agent=agents/gciql.py --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-medium-navigate-oraclerep-v0 (QRL)
python main.py --env_name=humanoidmaze-medium-navigate-oraclerep-v0 --agent=agents/qrl.py --agent.alpha=0.001 --agent.discount=0.995
# humanoidmaze-medium-navigate-oraclerep-v0 (CRL)
python main.py --env_name=humanoidmaze-medium-navigate-oraclerep-v0 --agent=agents/crl.py --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-medium-navigate-oraclerep-v0 (HIQL)
python main.py --env_name=humanoidmaze-medium-navigate-oraclerep-v0 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
# humanoidmaze-medium-navigate-oraclerep-v0 (TRL)
python main.py --env_name=humanoidmaze-medium-navigate-oraclerep-v0 --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.995 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0.0 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=0.1



# humanoidmaze-large-navigate-oraclerep-v0 (GCBC)
python main.py --env_name=humanoidmaze-large-navigate-oraclerep-v0 --agent=agents/gcbc.py
# humanoidmaze-large-navigate-oraclerep-v0 (GCIVL)
python main.py --env_name=humanoidmaze-large-navigate-oraclerep-v0 --agent=agents/gcivl.py --agent.alpha=10.0 --agent.discount=0.995
# humanoidmaze-large-navigate-oraclerep-v0 (GCIQL)
python main.py --env_name=humanoidmaze-large-navigate-oraclerep-v0 --agent=agents/gciql.py --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-large-navigate-oraclerep-v0 (QRL)
python main.py --env_name=humanoidmaze-large-navigate-oraclerep-v0 --agent=agents/qrl.py --agent.alpha=0.001 --agent.discount=0.995
# humanoidmaze-large-navigate-oraclerep-v0 (CRL)
python main.py --env_name=humanoidmaze-large-navigate-oraclerep-v0 --agent=agents/crl.py --agent.alpha=0.1 --agent.discount=0.995
# humanoidmaze-large-navigate-oraclerep-v0 (HIQL)
python main.py --env_name=humanoidmaze-large-navigate-oraclerep-v0 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
# humanoidmaze-large-navigate-oraclerep-v0 (TRL)
python main.py --env_name=humanoidmaze-large-navigate-oraclerep-v0 --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.995 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0.1 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=0.1


# pointmaze-medium-navigate-oraclerep-v0 (GCBC)
python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/gcbc.py
# pointmaze-medium-navigate-oraclerep-v0 (GCIVL)
python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/gcivl.py --agent.alpha=10.0
# pointmaze-medium-navigate-oraclerep-v0 (GCIQL)
python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/gciql.py --agent.alpha=0.003
# pointmaze-medium-navigate-oraclerep-v0 (QRL)
python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/qrl.py --agent.alpha=0.0003
# pointmaze-medium-navigate-oraclerep-v0 (CRL)
python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/crl.py --agent.alpha=0.03
# pointmaze-medium-navigate-oraclerep-v0 (HIQL)
python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0
# pointmaze-medium-navigate-oraclerep-v0 (TRL)
python main.py --env_name=pointmaze-medium-navigate-oraclerep-v0 --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(512, 512, 512)" --agent.value_hidden_dims="(512, 512, 512)" --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=True --agent.discount=0.99 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0.7 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=10



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
python main.py --env_name=puzzle-4x5-play-oraclerep-v0 --dataset_dir=<insert_dir> --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" --agent.discount=0.999 --dataset_replace_interval=1000 --agent.actor_geom_sample=True --agent.actor_p_trajgoal=0.5 --agent.actor_p_randomgoal=0.5 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=rejection



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
python main.py --env_name=puzzle-4x6-play-oraclerep-v0 --dataset_dir=<insert_dir> --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" --agent.discount=0.999 --dataset_replace_interval=1000 --agent.actor_geom_sample=True --agent.actor_p_trajgoal=0.5 --agent.actor_p_randomgoal=0.5 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=rejection


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
python main.py --env_name=humanoidmaze-giant-navigate-oraclerep-v0 --dataset_dir=<insert_dir> --agent=agents/trl.py --train_steps=1000000 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" --agent.discount=0.999 --dataset_replace_interval=4000 --agent.actor_geom_sample=False --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.value_geom_sample=False --agent.value_p_trajgoal=1.0 --agent.value_p_randomgoal=0.0 --agent.use_oracle_distillation=True --agent.expectile=0.7 --agent.distance_weight_lambda=0 --agent.policy_extraction=ddpgbc --agent.ddpgbc.alpha=0.1

