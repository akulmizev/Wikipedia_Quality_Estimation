#!bin/bash

sbatch -o pcm/pcm.raw_wiki.out pretrain_h100.slurm pcm raw_wiki 150
sbatch -o sw/sw.raw_wiki.out pretrain_h100.slurm sw raw_wiki 100

sbatch -o pcm/pcm.pre_filtered.out pretrain_h100.slurm pcm pre_filtered 150
sbatch -o sw/sw.pre_filtered.out pretrain_h100.slurm sw pre_filtered 100

sbatch -o pcm/pcm.length.high_quality.out pretrain_h100.slurm pcm length/high_quality 150
sbatch -o pcm/pcm.length.low_quality.out pretrain_h100.slurm pcm length/low_quality 150

sbatch -o sw/sw.length.high_quality.out pretrain_h100.slurm sw length/high_quality 100
sbatch -o sw/sw.length.low_quality.out pretrain_h100.slurm sw length/low_quality 100

sbatch -o pcm/pcm.unique_trigrams.high_quality.out pretrain_h100.slurm pcm unique_trigrams/high_quality 150
sbatch -o pcm/pcm.unique_trigrams.low_quality.out pretrain_h100.slurm pcm unique_trigrams/low_quality 150

sbatch -o sw/sw.unique_trigrams.high_quality.out pretrain_h100.slurm sw unique_trigrams/high_quality 100
sbatch -o sw/sw.unique_trigrams.low_quality.out pretrain_h100.slurm sw unique_trigrams/low_quality 100









