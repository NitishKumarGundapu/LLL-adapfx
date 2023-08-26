echo "................................................................................................................................"
echo " Training Adapter Fusion at rf 4" 
echo "................................................................................................................................"

python train_adapter_prefix_config.py \
    --data_dir /raid/amana/data \
    --model_dir_root models_ag \
    --seq_train_type lll --model_name gpt2 \
    --prefixlength 30 \
    --rf 4 \
    --learning_rate 0.001 \
    --bottle_neck_size 800 \
    --leaveout "" \
    --n_gpus 12 \
    --n_workers 75 \
    --fp32 \
    --seed 42 \
    --n_train_epochs 10 \
    --gen_lm_sample_percentage 0.2 \
    --tasks sst srl woz.en \
    --lm_lambda 0.2 \
    "$@"

echo "................................................................................................................................"
echo "Training Adapter + Prefix at prefix length 30"
echo "................................................................................................................................"

python test_adapter_from_scratch.py \
    --data_dir /raid/amana/data \
    --model_dir_root models_ag \
    --seq_train_type lll \
    --model_name gpt2 \
    --n_gpus 1 \
    --n_workers 35 \
    --n_train_epochs 10 \
    --gen_lm_sample_percentage 0.2 \
    --tasks sst srl woz.en \
    --lm_lambda 0.2 \
    "$@"
done 