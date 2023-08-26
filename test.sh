# #!/bin/bash -i

# source ./env

# python test.py \
#   --data_dir $DATA_DIR \
#   --model_dir_root $MODEL_ROOT_DIR \
#   "$@"

# python test_adapter_from_scratch.py --data_dir data --model_dir models --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 12 --gen_lm_sample_percentage 0.00 --tasks amazon wikisql sst srl woz.en --lm_lambda 0.0 > Testlogs_5aman

# python test_adapter_from_scratch.py --data_dir data --model_dir models --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 12 --gen_lm_sample_percentage 0.00 --tasks wikisql sst srl woz.en --lm_lambda 0.0 > Testlogs_4decaNLP

# nohup python test_adapter_from_scratch.py --data_dir data --model_dir models --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 2 --gen_lm_sample_percentage 0.05 --tasks sst srl woz.en --lm_lambda 0.1 > Testlogs_3og.out

# nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_ag --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 20 --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en --lm_lambda 0.2 > Testlogs_Adaptgen_3og.out

# nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_ccmd1 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 20 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0 > Testlogs_ccmd1_3og_preadap.out


# for n in ("1" "1 2" "1 2 3" "1 2 3 54")
# do 
#   echo $n
# done 

s=("1" "1,2" "1,2,3" "1,2,3,4,5")
for n in ${s[@]}; 
do
    echo $n
done