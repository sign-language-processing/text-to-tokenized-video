#!/bin/bash
# =============================================================================
# Text-to-Video Training Script
# =============================================================================
# Trains T5 models to translate German text to discrete video tokens.
#
# Usage:
#   ./script.sh t5-small-36    # Train T5-small for 36 epochs
#   ./script.sh t5-small-100   # Train T5-small for 100 epochs
#   ./script.sh t5-base        # Train T5-base for 36 epochs
# =============================================================================

set -euo pipefail

export WANDB_PROJECT=text2video

# Default configuration
MODEL="google-t5/t5-small"
EPOCHS=36
OUTPUT_DIR="models/text2video_t5small_36"

# Parse argument
case "${1:-t5-small-36}" in
    t5-small-36)
        MODEL="google-t5/t5-small"
        EPOCHS=36
        OUTPUT_DIR="models/text2video_t5small_36"
        ;;
    t5-small-100)
        MODEL="google-t5/t5-small"
        EPOCHS=100
        OUTPUT_DIR="models/text2video_t5small_100"
        ;;
    t5-base)
        MODEL="google-t5/t5-base"
        EPOCHS=36
        OUTPUT_DIR="models/text2video_t5base_36"
        ;;
    *)
        echo "Usage: $0 {t5-small-36|t5-small-100|t5-base}"
        exit 1
        ;;
esac

echo "=============================================="
echo "Text-to-Video Training"
echo "=============================================="
echo "Model:   $MODEL"
echo "Epochs:  $EPOCHS"
echo "Output:  $OUTPUT_DIR"
echo "=============================================="

python run_translation.py \
    --model_name_or_path "$MODEL" \
    --do_train \
    --do_eval \
    --do_predict \
    --source_lang en \
    --target_lang de \
    --train_file text2video/text_to_video_train.jsonl \
    --validation_file text2video/text_to_video_dev.jsonl \
    --test_file text2video/text_to_video_test.jsonl \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --max_source_length 512 \
    --max_target_length 2048 \
    --num_train_epochs "$EPOCHS" \
    --report_to wandb

echo ""
echo "Training complete! Model saved to: $OUTPUT_DIR"
