#!/bin/bash -i

source ./env

python train.py \
  --data_dir $DATA_DIR \
  --model_dir_root $MODEL_ROOT_DIR \
  "$@"

python train_experiment_adapter_from_scratch_mam_config.py --data_dir $DATA_DIR --model_dir_root $MODEL_ROOT_DIR --seq_train_type lll --model_name gpt2 --n_gpus 2 --n_workers 75 --fp32 --n_train_epochs 8 --gen_lm_sample_percentage 0.00 --tasks sst dbpedia srl amazon yelp woz.en --lm_lambda 0.0 > Trainlogs_without_replay_from_scratch_mam_config.out 2>&1&

nohup python train_experiment_adapter_from_scratch_mam_config.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 8 --n_workers 75 --fp32 --n_train_epochs 12 --gen_lm_sample_percentage 0.00 --tasks wikisql sst srl woz.en --lm_lambda 0.0 > Trainlogs_4decaNLP_mam_config.out

nohup python train_experiment_adapter_from_scratch_mam_config.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 8 --n_workers 75 --fp32 --n_train_epochs 12 --gen_lm_sample_percentage 0.00 --tasks wikisql amazon yelp --lm_lambda 0.0 > Trainlogs_3aman_mam_config.out

nohup python train_experiment_adapter_from_scratch_mam_config.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 8 --n_workers 75 --fp32 --n_train_epochs 12 --gen_lm_sample_percentage 0.00 --tasks amazon wikisql sst srl woz.en --lm_lambda 0.0 > Trainlogs_5aman_mam_config.out

nohup python train_experiment_adapter_from_scratch_mam_config.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 8 --n_workers 75 --fp32 --n_train_epochs 12 --gen_lm_sample_percentage 0.00 --tasks yahoo sst srl woz.en --lm_lambda 0.0 > Trainlogs_4cls_mam_config.out

nohup python train_experiment_adapter_from_scratch_mam_config.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 8 --n_workers 75 --fp32 --n_train_epochs 12 --gen_lm_sample_percentage 0.00 --tasks amazon dbpedia ag sst srl woz.en --lm_lambda 0.0 > Trainlogs_6cls_mam_config.out

nohup python train_experiment_adapter_from_scratch_mam_config.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 4 --n_workers 75 --fp32 --n_train_epochs 12 --gen_lm_sample_percentage 0.20 --tasks sst srl woz.en --lm_lambda 0.2 > Trainlogs_3og_gen_mam_config.out

nohup python train_adapter_gen_mam_config.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 4 --n_workers 75 --fp32 --n_train_epochs 2 --gen_lm_sample_percentage 0.05 --tasks sst srl woz.en --lm_lambda 0.1 > Trainlogs_ttadaptgen_3og_mam_config.out

nohup python train_adapter_gen_mam_config.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 10 --n_workers 75 --fp32 --n_train_epochs 8 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > Trainlogs_adaptgen_3og_mam_config.out

nohup python train_adapter_gen_mam_config.py --data_dir data --model_dir_root new_models --seq_train_type lll --model_name gpt2 --n_gpus 10 --n_workers 75 --fp32 --n_train_epochs 8 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > Trainlogs_AdaptGen_3og_mam_config.out

nohup python train_adapter_gen_mam_config.py --data_dir data --model_dir_root new_models --seq_train_type lll --model_name gpt2 --n_gpus 8 --n_workers 75 --fp32 --n_train_epochs 8 --gen_lm_sample_percentage 0.2 --tasks amazon ag yahoo yelp --lm_lambda 0.2 > Trainlogs_AdaptGen_4cls_mam_config.out

nohup python train_adapter_gen_mam_config.py --data_dir data --model_dir_root new_models --seq_train_type lll --model_name gpt2 --n_gpus 8 --n_workers 75 --fp32 --n_train_epochs 8 --gen_lm_sample_percentage 0.2 --tasks yahoo yelp amazon sst --lm_lambda 0.2 > Trainlogs_AdaptGen_4cls_mam_config.out

nohup python train_experiment_adapter_from_scratch_mam_config.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 8 --n_workers 75 --fp32 --n_train_epochs 8 --gen_lm_sample_percentage 0.00 --tasks dbpedia amazon ag sst srl woz.en --lm_lambda 0.0 > Trainlogs_6cls_mam_config.out

nohup python train_adapter_gen_mam_config.py --data_dir data --model_dir_root model_6f --seq_train_type lll --model_name gpt2 --n_gpus 2 --n_workers 75 --fp32 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks wikisql ag amazon sst srl woz.en --lm_lambda 0.2 > Trainlogs_adaptgen_6f.out

nohup python train_adapter_gen_mam_config.py --data_dir data --model_dir_root model_ag --seq_train_type lll --model_name gpt2 --n_gpus 2 --n_workers 75 --fp32 --n_train_epochs 20 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > Trainlogs_AdaptGen_3og.out

nohup python train.py --data_dir data --model_dir_root model_ag --seq_train_type lll --model_name gpt2 --n_gpus 2 --n_workers 75 --fp32 --n_train_epochs 20 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > Trainlogs_lamol_3og.out

nohup python train_adapter_gen_mam_config.py --data_dir data --model_dir_root model_ccmd1 --seq_train_type lll --model_name gpt2 --n_gpus 2 --n_workers 75 --fp32 --n_train_epochs 20 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_ccmd1_3og_preadap.out