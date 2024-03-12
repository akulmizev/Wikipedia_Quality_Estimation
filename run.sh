#!/bin/bash

source activate hf

config_file="./config/as/config.json"
tokenizer_file="./tokenizers/wiki.as.json"

python run_mlm.py \
  --model_type roberta \
  --config_name $config_file \
  --tokenizer_name $tokenizer_file \
  --dataset_name 'WikiQuality/as.filtered' \
  --validation_split_percentage 5 \

