#!/usr/bin/env python3

import csv
import json
import os

import torch
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================

BASE_PT_DIR = os.path.expanduser("~/data/thesis_storage/tokens")
BASE_CSV_DIR = os.path.expanduser("~/scratch/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual")

OUTPUT_DIR = ".."
os.makedirs(OUTPUT_DIR, exist_ok=True)

CORRUPTED_TEST_FILES = {
    "20June_2011_Monday_tagesschau-1249.pt",
    "30December_2010_Thursday_tagesschau-2232.pt",
    "30October_2009_Friday_tagesschau-2273.pt",
    "10February_2010_Wednesday_tagesschau-2515.pt",
    "14December_2009_Monday_tagesschau-2092.pt",
    "25August_2010_Wednesday_tagesschau-8516.pt",
    "27August_2009_Thursday_tagesschau-3269.pt",
    "06February_2010_Saturday_tagesschau-7365.pt",
}

SPLITS = {
    "train": {
        "pt_dir": os.path.join(BASE_PT_DIR, "train"),
        "csv": "PHOENIX-2014-T.train.corpus.csv",
        "out": "gloss_to_video_train.jsonl",
    },
    "dev": {
        "pt_dir": os.path.join(BASE_PT_DIR, "dev"),
        "csv": "PHOENIX-2014-T.dev.corpus.csv",
        "out": "gloss_to_video_dev.jsonl",
    },
    "test": {
        "pt_dir": os.path.join(BASE_PT_DIR, "test"),
        "csv": "PHOENIX-2014-T.test.corpus.csv",
        "out": "gloss_to_video_test.jsonl",
    },
}

# =========================
# LOAD GLOSS ANNOTATIONS
# =========================


def load_gloss_map(csv_path):
    gloss_map = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            gloss_map[row["name"]] = row["orth"].split()
    return gloss_map


# =========================
# BUILD DATASET
# =========================


def build_dataset(pt_dir, gloss_map, skip_files=None):
    dataset = []

    for fname in tqdm(sorted(os.listdir(pt_dir))):
        if not fname.endswith(".pt"):
            continue

        if skip_files and fname in skip_files:
            continue

        video_id = fname.replace(".pt", "")
        if video_id not in gloss_map:
            continue

        pt_path = os.path.join(pt_dir, fname)
        tokens = torch.load(pt_path)[0].flatten().tolist()

        entry = {
            "translation": {
                # SOURCE = GLOSS
                "en": " ".join(gloss_map[video_id]),
                # TARGET = VIDEO TOKENS
                "de": " ".join(map(str, tokens)),
            }
        }

        dataset.append(entry)

    return dataset


# =========================
# SAVE JSONL
# =========================


def save_jsonl(data, path):
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
        pt_dir = cfg["pt_dir"]
        out_path = os.path.join(OUTPUT_DIR, cfg["out"])

        gloss_map = load_gloss_map(csv_path)

        skip = CORRUPTED_TEST_FILES if split == "test" else None
        dataset = build_dataset(pt_dir, gloss_map, skip_files=skip)

        save_jsonl(dataset, out_path)

        print(f"✅ Saved {len(dataset)} samples → {out_path}")
