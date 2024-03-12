#!/bin/bash

conda activate hf

config_file="./config/as/config.json"

python run_mlm.py \
  --model_type roberta \
  --config_name $config_file \
  --tokenizer_name "WikiQuality/as_tokenizer" \
  --dataset_name 'WikiQuality/as.filtered' \
  --validation_split_percentage 5 \
  --output_dir './as_filtered' \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \