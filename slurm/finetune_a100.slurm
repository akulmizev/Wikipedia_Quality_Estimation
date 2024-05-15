#!/bin/bash
#SBATCH --clusters=wice
#SBATCH -A lp_lagom
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=kushaljayesh.tatariya@kuleuven.be
#SBATCH --job-name=wiki

cd /data/leuven/351/vsc35188/miniconda3/bin/
source activate wiki

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vsc-hard-mounts/leuven-data/351/vsc35188/miniconda3/envs/wiki/lib

cd /lustre1/project/stg_00120/kushal/Wikipedia/Wikipedia_Quality_Estimation/

lang=$1
partition=$2 #pass <partition>/high_quality, <partition>/low_quality for the partitions
task=$3

config=config/experiment/pretrain/$partition/$task.yaml

python wqe/run_experiment.py --config $config --wiki_id $lang