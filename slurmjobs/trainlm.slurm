#!/bin/bash
#SBATCH --job-name=smollm3-signlang
#SBATCH --output=/home/mpanag/thesis/text-to-tokenized-video/slurmjobs/logs/301smollm_%j.out
#SBATCH --error=/home/mpanag/thesis/text-to-tokenized-video/slurmjobs/logs/301smollm_%j.err
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

CONFIG_PATH=~/thesis/text-to-tokenized-video/slurmjobs/config_smollm301.yaml
CHECKPOINT_DIR=~/thesis/text-to-tokenized-video/checkpoints/smollm2-signlang-v301

# === CREATE OUTPUT DIR AND CHANGE TO PARENT ===
# autotrain creates {project_name} folder in the current working directory
# We cd to the parent of CHECKPOINT_DIR so that the project_name folder is created there
mkdir -p "$(dirname "$CHECKPOINT_DIR")"
cd "$(dirname "$CHECKPOINT_DIR")"
echo "Working directory: $(pwd)"

# === RUN TRAINING ===
# autotrain will create a folder named after project_name (smollm2-signlang-v301) in current dir
autotrain --config "$CONFIG_PATH"

# === FORCE SAVE IF NOT DONE ===
echo "=== Checking for saved model ==="

# List what was created
echo "Contents of checkpoint directory:"
ls -la "$CHECKPOINT_DIR" 2>/dev/null || echo "Checkpoint directory does not exist yet"

# autotrain may save different file formats (safetensors, pytorch_model.bin, etc.)
SAVED_MODEL=$(find "$CHECKPOINT_DIR" -type f \( -name "*.safetensors" -o -name "pytorch_model.bin" \) 2>/dev/null | head -n 1)

if [ -n "$SAVED_MODEL" ]; then
    echo "✓ Found saved model at: $SAVED_MODEL"
else
    echo "⚠️ WARNING: No saved model found in $CHECKPOINT_DIR"
    echo "Checking current directory for model files..."
    find . -type f \( -name "*.safetensors" -o -name "pytorch_model.bin" \) 2>/dev/null | head -5
fi

echo "=== Training Complete ==="
date