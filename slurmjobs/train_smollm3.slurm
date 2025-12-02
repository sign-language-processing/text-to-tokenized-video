#!/bin/bash
#SBATCH --job-name=smollm3-signlang
#SBATCH --output=/home/mpanag/thesis/text-to-tokenized-video/slurmjobs/logs/smollm3_%j.out
#SBATCH --error=/home/mpanag/thesis/text-to-tokenized-video/slurmjobs/logs/smollm3_%j.err
#SBATCH --time=24:00:00
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

EXPERIMENT_DIR=~/scratch/lm_storage/experiments/2025-12-01_smol135m_sft_5ep
CHECKPOINT_DIR=$EXPERIMENT_DIR/smollm2-signlang-v3

mkdir -p "$CHECKPOINT_DIR"
cd "$EXPERIMENT_DIR"

echo "=== Starting SmolLM2 v3 Fine-tuning (Scratch Save) ==="
date

# === RUN TRAINING ===
autotrain --config ~/thesis/text-to-tokenized-video/slurmjobs/config_smollm3.yaml

echo "=== Training Complete ==="
date
