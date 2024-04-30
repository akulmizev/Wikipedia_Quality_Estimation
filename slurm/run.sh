#!/bin/bash

source activate hf

export WANDB_API_KEY="2f9506e16930f137abbd18a3fb16f6b31840a830"
export WANDB_PROJECT="WikiQuality"

lang=$1
partition=$2

config_file="./config/$lang/config_deberta.json"

python scripts/run_mlm.py \
  --model_type "deberta" \
  --config_name $config_file \
  --tokenizer_name "WikiQuality/$partition.$lang" \
  --dataset_name 'WikiQuality/' \
  --output_dir "$lang\_$partition\_$epochs\_8_tiny_deberta" \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --mlm_probability 0.4 \
  --overwrite_output_dir \
  --overwrite_cache \
  --pad_to_max_length \
  --line_by_line \
  --report_to 'wandb' \
  --run_name "$lang\_$partition\_$epochs\_8_tiny_deberta" \
  --num_train_epochs $epochs \
  --save_strategy 'epoch' \
  --evaluation_strategy 'epoch' \
  --save_total_limit 5 \
  --load_best_model_at_end \
  --metric_for_best_model 'eval_loss'