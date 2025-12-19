#!/usr/bin/env python3
"""
Gloss-to-Text Dataset Builder

Creates JSONL files for training T5 to translate sign language glosses to German text.
Source (en): Gloss sequence (e.g., "WETTER MORGEN REGEN")
Target (de): German translation (e.g., "Das Wetter morgen wird regnerisch")
"""

import csv
import json
import os

# =========================
# CONFIGURATION
# =========================

BASE_CSV_DIR = os.path.expanduser("~/scratch/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual")

OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLITS = {
    "train": {
        "csv": "PHOENIX-2014-T.train.corpus.csv",
        "out": "gloss_to_text_train.jsonl",
    },
    "dev": {
        "csv": "PHOENIX-2014-T.dev.corpus.csv",
        "out": "gloss_to_text_dev.jsonl",
    },
    "test": {
        "csv": "PHOENIX-2014-T.test.corpus.csv",
        "out": "gloss_to_text_test.jsonl",
    },
}

# =========================
# BUILD DATASET
# =========================


def build_dataset(csv_path):
    """
    Read PHOENIX CSV and create gloss -> text translation pairs.

    Args:
        csv_path: Path to PHOENIX corpus CSV file

    Returns:
        List of translation entries in HuggingFace format
    """
    dataset = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            gloss = row["orth"].strip()
            text = row["translation"].strip()

            entry = {
                "translation": {
                    "en": gloss,  # SOURCE = GLOSS
                    "de": text,  # TARGET = GERMAN TEXT
                }
            }
            dataset.append(entry)

    return dataset


# =========================
# SAVE JSONL
# =========================


def save_jsonl(data, path):
    """Save dataset as JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    for split, cfg in SPLITS.items():
        print(f"\n📂 Processing {split}")

        csv_path = os.path.join(BASE_CSV_DIR, cfg["csv"])
        out_path = os.path.join(OUTPUT_DIR, cfg["out"])

        if not os.path.exists(csv_path):
            print(f"⚠️  CSV not found: {csv_path}")
            continue

        dataset = build_dataset(csv_path)
        save_jsonl(dataset, out_path)

        print(f"✅ Saved {len(dataset)} samples → {out_path}")
