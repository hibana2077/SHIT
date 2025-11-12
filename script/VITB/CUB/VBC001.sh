#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=24GB
#PBS -l walltime=08:00:00
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