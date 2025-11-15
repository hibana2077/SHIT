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
    --model-name tinynet_c.in1k \
    --dataset-name cub_200_2011 \
    --batch-size 32 \
    --num-epochs 200 \
    --lr 1e-4 \
    --output-dir ./outputs/TNCC001 --num-classes 200 --val-split test >> TNCC001.log