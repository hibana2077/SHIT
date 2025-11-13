#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=24GB
#PBS -l walltime=28:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/SHIT/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/SHIT/.cache"
export HF_HUB_OFFLINE=1

cd ../../..
python3 -m src.train \
  --dataset-name cub_200_2011 \
  --data-root ./data \
  --model-name vit_small_r26_s32_384.augreg_in21k_ft_in1k \
  --head custom \
  --custom-head-module src.head.outlier_impute_head \
  --custom-head-class OutlierImputeHead \
  --custom-head-kwargs '{"n":512,"top_k":1}' \
  --batch-size 32 \
  --output-dir ./outputs/VSC006 --num-classes 200 --val-split test --lr 1e-4 >> VSC006.log