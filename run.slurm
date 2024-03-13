#!/bin/bash
#SBATCH --clusters=genius
#SBATCH -A lp_lagom
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_v100
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=kushaljayesh.tatariya@kuleuven.be
#SBATCH --job-name=train_tinybert


cd /data/leuven/351/vsc35188/miniconda3/bin/
source activate hf

cd $SLURM_SUBMIT_DIR

export WANDB_API_KEY="2f9506e16930f137abbd18a3fb16f6b31840a830"
export WANDB_PROJECT="WikiQuality"

hub_token=$(cat huggingface_token.txt)

config_file="./config/as/config.json"

python run_mlm.py \
  --model_type roberta \
  --config_name "roberta" \
  --config_overrides $config_file \
  --tokenizer_name "WikiQuality/as_tokenizer" \
  --dataset_name 'WikiQuality/as.filtered' \
  --validation_split_percentage 5 \
  --output_dir 'as_filtered' \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 256 \
  --per_device_eval_batch_size 8 \
  --overwrite_output_dir \
  --overwrite_cache \
  --pad_to_max_length \
  --line_by_line \
  --report_to 'wandb' \
  --run_name 'as_filtered' \
  --num_train_epochs 5 \
  --push_to_hub \
  --hub_model_id 'tiny_bert_as_filtered' \
  --hub_private_repo \
  --hub_token $hub_token