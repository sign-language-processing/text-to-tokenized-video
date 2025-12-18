#!/bin/bash
# Gloss-to-Text Training Script
# Translates sign language glosses to German text using T5

export WANDB_PROJECT=gloss2text

python run_translation.py \
  --model_name_or_path google-t5/t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --source_lang en \
  --target_lang de \
  --train_file gloss2text/gloss_to_text_train.jsonl \
  --validation_file gloss2text/gloss_to_text_dev.jsonl \
  --test_file gloss2text/gloss_to_text_test.jsonl \
  --output_dir models/gloss2text \
  --per_device_train_batch_size=4 \
  --per_device_eval_batch_size=4 \
  --predict_with_generate \
  --max_source_length 512 \
  --num_train_epochs 36 \
  --report_to wandb
