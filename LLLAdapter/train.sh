#!/bin/bash -i

source ./env

python train_experiment_adapter_from_scratch_mam_config.py \
  --data_dir $DATA_DIR \
  --model_dir_root $MODEL_ROOT_DIR \
  "$@"
