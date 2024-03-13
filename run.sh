#!/bin/bash

export WANDB_API_KEY="2f9506e16930f137abbd18a3fb16f6b31840a830"
export WANDB_PROJECT="WikiQuality"

config_file="./config/as/config.json"

python run_mlm.py \
  --model_type roberta \
  --tokenizer_name "WikiQuality/as_tokenizer" \
  --dataset_name 'WikiQuality/as.filtered' \
  --validation_split_percentage 5 \
  --output_dir 'as_filtered' \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --overwrite_output_dir \
  --report_to 'wandb' \
  --overwrite_cache \
  --pad_to_max_length \
  --line_by_line
