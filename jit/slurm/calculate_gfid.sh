#!/bin/bash
#SBATCH --job-name=jit_gfid
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/jit_gfid_%j.out

# Usage (from jit/):
#   sbatch slurm/calculate_gfid.sh <model_dir> <reference_dir> <flags of main_jit.py>
# <model_dir> holds checkpoint-last.pth. Use the same dataset flags as in training.
set -e
OUT=$(realpath "$1")
REF=$(realpath "$2")
shift 2
NUM_SAMPLES=${NUM_SAMPLES:-15000}

mkdir -p logs
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --master_port=$(( 20000 + RANDOM % 40000 )) \
    main_jit.py \
    --img_size 256 --noise_scale 1.0 \
    --P_mean -0.8 --P_std 0.8 \
    --proj_dropout 0.0 \
    --batch_size 32 --num_workers 4 \
    --epochs 1 --warmup_epochs 1 \
    --evaluate_gen \
    --num_images "$NUM_SAMPLES" --gen_bsz 128 \
    --num_sampling_steps 50 --sampling_method heun \
    --cfg 1.0 --interval_min 0.1 --interval_max 1.0 \
    --output_dir "$OUT" --resume "$OUT" \
    "$@"

# main_jit.py writes the images into a sub-folder of the output directory.
SAMPLES=$(find "$OUT" -name '*.png' -printf '%h\n' | sort -u)
if [ "$(echo "$SAMPLES" | wc -l)" != 1 ]; then
    echo "ERROR: expected one sample folder in $OUT, found: $SAMPLES"
    exit 1
fi

cd ..
python -m dgm_eval "$REF" "$SAMPLES" \
    --model inception \
    --metrics fd kd prdc \
    --batch_size 1024 \
    --nsample "$NUM_SAMPLES" \
    --avg-all-channels \
    --no-load
