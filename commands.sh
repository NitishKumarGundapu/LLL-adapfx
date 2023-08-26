s=("0,1,2,3,4,5,6,7,8,9,10" "0,1,2,3,4,5,6,7" "0,1,2,3,4,5" "0,1,2,3" "4,5,6,7,8,9,10,11" "6,7,8,9,10,11" "8,9,10,11" "0,2,4,6,8,10" "1,3,5,7,9,11")
for n in ${s[@]}; 
do
    echo "................................................................................................................................"
    echo " Training Adapters at rf 4" 
    echo "................................................................................................................................"

    python train_adapter_config.py \
        --data_dir data \
        --model_dir_root models_ag \
        --seq_train_type lll --model_name gpt2 \
        --prefixlength 30 \
        --rf 4 \
        --flat True \
        --bottle_neck_size 800 \
        --leaveout $n \
        --n_gpus 12 \
        --n_workers 75 \
        --fp32 \
        --seed 42 \
        --n_train_epochs 20 \
        --gen_lm_sample_percentage 0.2 \
        --tasks sst srl woz.en \
        --lm_lambda 0.2 \
        "$@"

    echo "................................................................................................................................"
    echo "Testing Adapters at rf 4" 
    echo "................................................................................................................................"

    python test_adapter_from_scratch.py \
        --data_dir data \
        --model_dir_root models_ag \
        --seq_train_type lll \
        --model_name gpt2 \
        --n_gpus 1 \
        --n_workers 35 \
        --n_train_epochs 20 \
        --gen_lm_sample_percentage 0.2 \
        --tasks sst srl woz.en \
        --lm_lambda 0.2 \
        "$@"
done 

echo "................................................................................................................................"
echo " Training Adapters at rf 4" 
echo "................................................................................................................................"

python train_adapter_config.py \
    --data_dir data \
    --model_dir_root models_ag \
    --seq_train_type lll --model_name gpt2 \
    --prefixlength 30 \
    --rf 4 \
    --flat True \
    --bottle_neck_size 800 \
    --leaveout "" \
    --n_gpus 12 \
    --n_workers 75 \
    --fp32 \
    --seed 42 \
    --n_train_epochs 20 \
    --gen_lm_sample_percentage 0.2 \
    --tasks sst srl woz.en \
    --lm_lambda 0.2 \
    "$@"

echo "................................................................................................................................"
echo "Testing Adapters at rf 4" 
echo "................................................................................................................................"

python test_adapter_from_scratch.py \
    --data_dir data \
    --model_dir_root models_ag \
    --seq_train_type lll \
    --model_name gpt2 \
    --n_gpus 1 \
    --n_workers 35 \
    --n_train_epochs 20 \
    --gen_lm_sample_percentage 0.2 \
    --tasks sst srl woz.en \
    --lm_lambda 0.2 \
    "$@"

echo "--------------------------------------------The End Man!!!------------------------------------------------------" 