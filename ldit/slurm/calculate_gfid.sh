#!/bin/bash
#SBATCH --job-name=ldit_gfid
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/ldit_gfid_%j.out

# Usage (from ldit/):
#   sbatch slurm/calculate_gfid.sh configs/<inference_config>.yaml <ckpt> <reference_dir> <output_dir>
set -e
CONFIG=$1
CKPT=$2
REF=$(realpath "$3")
OUT=$(realpath -m "$4")
NGPUS=2
NUM_SAMPLES=${NUM_SAMPLES:-15000}

mkdir -p logs
accelerate launch \
    --mixed_precision=bf16 \
    --num_processes=${NGPUS} \
    inference.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --output-dir "$OUT" \
    --num-samples "$NUM_SAMPLES" \
    --seed 42

# inference.py writes the images into a sub-folder of the output directory.
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
