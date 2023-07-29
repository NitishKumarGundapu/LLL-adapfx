#see sample test command

#1
nohup python test_adapter_from_scratch.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 35 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_testing_logs/Testlogs_adapter_prefix_last_1_layer.out
#4
nohup python test_adapter_from_scratch.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 35 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_testing_logs/Testlogs_adapter_prefix_last_4_layer.out
#6
nohup python test_adapter_from_scratch.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 35 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_testing_logs/Testlogs_adapter_prefix_last_6_layer.out
#8
nohup python test_adapter_from_scratch.py --data_dir data --model_dir_root models_ag --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 35 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_testing_logs/Testlogs_adapter_prefix_last_8_layer.out
#full
nohup python test_adapter_from_scratch.py --data_dir data --model_dir_root models_ag --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 35 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_testing_logs/Testlogs_adapter_prefix_last_full_layer.out
#alternate_even
nohup python test_adapter_from_scratch.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 35 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_testing_logs/Testlogs_adapter_prefix_last_alternate_layer.out

#alternate_odd
nohup python test_adapter_from_scratch.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 35 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_testing_logs/Testlogs_adapter_prefix_last_alternate_odd_layer.out
#first_4
nohup python test_adapter_from_scratch.py --data_dir data --model_dir_root models_ag --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 35 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_testing_logs/Testlogs_adapter_prefix_first_4_layer.out
#first_6
nohup python test_adapter_from_scratch_1.py --data_dir data --model_dir_root model_ag --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 35 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_testing_logs/Testlogs_adapter_prefix_first_6_layer.out
#first_8
nohup python test_adapter_from_scratch.py --data_dir data --model_dir_root new_models --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 35 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_testing_logs/Testlogs_adapter_prefix_first_8_layer.out



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#sample train command

#1
nohup python train_adapter_gen_mam_config_2.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 12 --n_workers 75 --fp32 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_training_logs/Trainlogs_adapter_prefix_last_1_layer.out
#4
nohup python train_adapter_gen_mam_config_2.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 12 --n_workers 75 --fp32 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_training_logs/Trainlogs_adapter_prefix_last_4_layer.out
#6
nohup python train_adapter_gen_mam_config_2.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 12 --n_workers 75 --fp32 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_training_logs/Trainlogs_adapter_prefix_last_6_layer.out
#8
nohup python train_adapter_gen_config.py --data_dir data --model_dir_root models_ag --seq_train_type lll --model_name gpt2 --n_gpus 12 --n_workers 75 --fp32 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_training_logs/Trainlogs_adapter_prefix_last_8_layer.out
#full
nohup python train_adapter_gen_config.py --data_dir data --model_dir_root models_ag --seq_train_type lll --model_name gpt2 --n_gpus 12 --n_workers 75 --fp32 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_training_logs/Trainlogs_adapter_prefix_last_full_layer.out
#alternate_even
nohup python train_adapter_gen_mam_config_2.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 12 --n_workers 75 --fp32 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_training_logs/Trainlogs_adapter_prefix_last_alternate_layer.out

#alternate_odd
nohup python train_adapter_prefix_config_3.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 12 --n_workers 75 --fp32 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_training_logs/Trainlogs_adapter_prefix_last_alternate_odd_layer.out
#first_4
nohup python train_adapter_prefix_config_0.py --data_dir data --model_dir_root models_ag --seq_train_type lll --model_name gpt2 --n_gpus 12 --n_workers 75 --fp32 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_training_logs/Trainlogs_adapter_prefix_first_4_layer.out
#first_6
nohup python train_adapter_prefix_config_1.py --data_dir data --model_dir_root model_ag --seq_train_type lll --model_name gpt2 --n_gpus 12 --n_workers 75 --fp32 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_training_logs/Trainlogs_adapter_prefix_first_6_layer.out
#first_8
nohup python train_adapter_prefix_config_2.py --data_dir data --model_dir_root new_models --seq_train_type lll --model_name gpt2 --n_gpus 12 --n_workers 75 --fp32 --n_train_epochs 10 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_training_logs/Trainlogs_adapter_prefix_first_8_layer.out
