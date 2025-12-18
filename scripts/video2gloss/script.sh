#!/bin/bash
# =============================================================================
# Video-to-Gloss Training Script
# =============================================================================
# Trains T5 to translate discrete video tokens to sign language glosses.
#
# Usage:
#   ./script.sh              # Train with default settings (T5-small)
#   ./script.sh t5-base      # Train with T5-base
# =============================================================================

set -euo pipefail

export WANDB_PROJECT=video2gloss

# Default configuration
MODEL="google-t5/t5-small"
OUTPUT_DIR="models/video2gloss"

# Parse argument
case "${1:-t5-small}" in
    t5-small)
        MODEL="google-t5/t5-small"
        OUTPUT_DIR="models/video2gloss_t5small"
        ;;
    t5-base)
        MODEL="google-t5/t5-base"
        OUTPUT_DIR="models/video2gloss_t5base"
        ;;
    *)
        echo "Usage: $0 {t5-small|t5-base}"
        exit 1
        ;;
esac

echo "=============================================="
echo "Video-to-Gloss Training"
echo "=============================================="
echo "Model:   $MODEL"
echo "Output:  $OUTPUT_DIR"
echo "=============================================="

python run_translation.py \
    --model_name_or_path "$MODEL" \
    --do_train \
    --do_eval \
    --do_predict \
    --source_lang en \
    --target_lang de \
    --train_file video2gloss/video_to_gloss_train.jsonl \
    --validation_file video2gloss/video_to_gloss_dev.jsonl \
    --test_file video2gloss/video_to_gloss_test.jsonl \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --max_source_length 2048 \
    --max_target_length 512 \
    --num_train_epochs 36 \
    --report_to wandb

echo ""
echo "Training complete! Model saved to: $OUTPUT_DIR"
