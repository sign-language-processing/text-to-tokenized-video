#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/encoder.jit"
SPLITS=("train" "dev" "test")

mkdir -p "$TOKENS_DIR"

for split in "${SPLITS[@]}"; do
  find "$PHOENIX/$split" -mindepth 1 -maxdepth 1 -type d | while read -r seq_dir; do
    seq_id=$(basename "$seq_dir")
    out_path="$TOKENS_DIR/$split/$seq_id.pt"
    mkdir -p "$(dirname "$out_path")"

    # Skip if already exists
    if [[ -f "$out_path" ]]; then
      echo "Skipping $split/$seq_id (already exists)"
      continue
    fi

    echo "Encoding $split/$seq_id..."
    python -m text_to_tokenized_video.tokenizer.encode_video \
      --video="$seq_dir" \
      --checkpoint-enc="$CHECKPOINT" \
      --output-path="$out_path" &

    # Limit concurrency
    if (( $(jobs -r | wc -l) >= 8 )); then
      wait -n
    fi
  done
done

wait
echo "✅ All done."
