#!/bin/bash
#SBATCH --job-name=jit_train
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/jit_train_%j.out

# Usage (from jit/): sbatch slurm/train.sh <output_dir> <flags of main_jit.py>
# See README.md for the flags of each dataset.
set -e
OUT=$1
shift

mkdir -p logs "$OUT"
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --master_port=$(( 20000 + RANDOM % 40000 )) \
    main_jit.py \
    --img_size 256 --noise_scale 1.0 \
    --P_mean -0.8 --P_std 0.8 \
    --proj_dropout 0.0 \
    --batch_size 144 --blr 5e-5 \
    --num_workers 16 \
    --warmup_epochs 5 \
    --save_last_freq 5 --save_persistent_freq 20 \
    --interval_min 0.1 --interval_max 1.0 \
    --output_dir "$OUT" --resume "$OUT" \
    "$@"
