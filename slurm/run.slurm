#!/bin/bash
#SBATCH --clusters=genius
#SBATCH -A lp_lagom
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_v100
#SBATCH --time=09:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=kushaljayesh.tatariya@kuleuven.be
#SBATCH --job-name=train_tinybert


cd /data/leuven/351/vsc35188/miniconda3/bin/
source activate hf

cd $SLURM_SUBMIT_DIR

export WANDB_API_KEY="2f9506e16930f137abbd18a3fb16f6b31840a830"
export WANDB_PROJECT="WikiQuality"

lang=$1
partition=$2
epochs=$3

#hub_token=$(cat huggingface_token.txt)
python create_model_config.py --lang $lang
python create_text_file.py --lang $lang --partition $partition

config_file="./config/$lang/config_tiny_deberta/config.json"

python run_mlm.py \
  --model_type "deberta" \
  --config_name $config_file \
  --tokenizer_name "WikiQuality/$lang\_tokenizer" \
  --dataset_name 'data/as/' \
  --validation_split_percentage 5 \
  --output_dir "$lang\_$partition\_$epochs\_8_tiny_deberta" \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
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

