#!/bin/bash
#SBATCH --job-name=smollm3-signlang
#SBATCH --output=/home/mpanag/thesis/text-to-tokenized-video/slurmjobs/logs/3smollm_%j.out
#SBATCH --error=/home/mpanag/thesis/text-to-tokenized-video/slurmjobs/logs/3smollm_%j.err
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=standard
#SBATCH --ntasks=1

# === ENVIRONMENT SETUP ===
source ~/thesis/cosmos-env/bin/activate

export TMPDIR=/home/mpanag/thesis/text-to-tokenized-video/tmp
export HF_HOME=/home/mpanag/thesis/text-to-tokenized-video/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $TMPDIR $HF_HOME

echo "=== Starting SmolLM2 v3 Fine-tuning ==="
date

CONFIG_PATH=~/thesis/text-to-tokenized-video/slurmjobs/config_smollm3.yaml
CHECKPOINT_DIR=~/thesis/text-to-tokenized-video/checkpoints/smollm2-signlang-v3

# === FORCE MODEL SAVE PATH ===
export AUTOTRAIN_SAVE_PATH="$CHECKPOINT_DIR"

# === RUN TRAINING ===
autotrain --config "$CONFIG_PATH"

# === FORCE SAVE IF NOT DONE ===
echo "=== Checking for saved model ==="
SAVED_MODEL=$(find . -type f -name "pytorch_model.bin" | head -n 1)

if [ -n "$SAVED_MODEL" ]; then
    echo "Found saved model at: $SAVED_MODEL"
    echo "Copying model files to: $CHECKPOINT_DIR"
    cp $(dirname $SAVED_MODEL)/* "$CHECKPOINT_DIR"/
else
    echo "⚠️ WARNING: No saved model found! Training may not have saved anything."
fi

echo "=== Training Complete ==="
date
