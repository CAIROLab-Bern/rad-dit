#!/bin/bash
#SBATCH --job-name=ldit_train
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/ldit_train_%j.out

# Usage (from ldit/): sbatch slurm/train.sh configs/<training_config>.yaml
set -e
CONFIG=$1
NGPUS=2

mkdir -p logs
accelerate launch \
    --mixed_precision=bf16 \
    --num_processes=${NGPUS} \
    train_shards.py \
    --config "$CONFIG" "${@:2}"
