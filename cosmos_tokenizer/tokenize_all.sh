#!/usr/bin/env bash
# =============================================================================
# Tokenize All PHOENIX Videos
# =============================================================================
# Encodes all sign language videos from the PHOENIX-2014-T dataset into
# discrete video tokens using the Cosmos DV8x16x16 tokenizer.
#
# Usage:
#   ./tokenize_all.sh
#
# Environment variables:
#   PHOENIX_DIR   - Path to PHOENIX dataset features (default: ~/scratch/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px)
#   TOKENS_DIR    - Output directory for tokens (default: ~/data/thesis_storage/tokens)
#   CHECKPOINT    - Path to encoder checkpoint (default: checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/encoder.jit)
#   MAX_PARALLEL  - Maximum parallel encoding jobs (default: 8)
# =============================================================================

set -euo pipefail

# Configuration
PHOENIX_DIR="${PHOENIX_DIR:-$HOME/scratch/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px}"
TOKENS_DIR="${TOKENS_DIR:-$HOME/data/thesis_storage/tokens}"
CHECKPOINT="${CHECKPOINT:-checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/encoder.jit}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"

SPLITS=("train" "dev" "test")

echo "=============================================="
echo "PHOENIX Video Tokenization"
echo "=============================================="
echo "Source:     $PHOENIX_DIR"
echo "Output:     $TOKENS_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Parallel:   $MAX_PARALLEL jobs"
echo "=============================================="

# Verify checkpoint exists
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Encoder checkpoint not found: $CHECKPOINT"
    echo "Run 'make download-checkpoints' first."
    exit 1
fi

# Create output directories
mkdir -p "$TOKENS_DIR"

for split in "${SPLITS[@]}"; do
    echo ""
    echo "Processing $split split..."

    split_dir="$PHOENIX_DIR/$split"
    out_dir="$TOKENS_DIR/$split"
    mkdir -p "$out_dir"

    if [[ ! -d "$split_dir" ]]; then
        echo "WARNING: Split directory not found: $split_dir"
        continue
    fi

    # Find all video sequence directories
    find "$split_dir" -mindepth 1 -maxdepth 1 -type d | while read -r seq_dir; do
        seq_id=$(basename "$seq_dir")
        out_path="$out_dir/$seq_id.pt"

        # Skip if already tokenized
        if [[ -f "$out_path" ]]; then
            echo "  [SKIP] $seq_id (already exists)"
            continue
        fi

        echo "  [ENCODE] $seq_id..."
        python cosmos_tokenizer/cosmos_tokenizer.py encode \
            --video "$seq_dir" \
            --checkpoint-enc "$CHECKPOINT" \
            --output "$out_path" &

        # Limit parallel jobs
        if (( $(jobs -r | wc -l) >= MAX_PARALLEL )); then
            wait -n
        fi
    done

    # Wait for split to complete
    wait
    echo "Completed $split split."
done

echo ""
echo "=============================================="
echo "Tokenization complete!"
echo "Tokens saved to: $TOKENS_DIR"
echo "=============================================="
