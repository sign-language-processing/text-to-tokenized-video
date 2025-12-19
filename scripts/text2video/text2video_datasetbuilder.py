#!/usr/bin/env python3
import csv
import json
import os

import torch
from tqdm import tqdm

# === CONFIGURATION ===
BASE_PT_DIR = os.path.expanduser("~/data/thesis_storage/tokens")
BASE_CSV_PATH = os.path.expanduser("~/scratch/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual")

SPLITS = {
    "train": {
        "pt_dir": os.path.join(BASE_PT_DIR, "train"),
        "csv_path": os.path.join(BASE_CSV_PATH, "PHOENIX-2014-T.train.corpus.csv"),
        "output": "text_to_video_train.jsonl",
    },
    "dev": {
        "pt_dir": os.path.join(BASE_PT_DIR, "dev"),
        "csv_path": os.path.join(BASE_CSV_PATH, "PHOENIX-2014-T.dev.corpus.csv"),
        "output": "text_to_video_dev.jsonl",
    },
    "test": {
        "pt_dir": os.path.join(BASE_PT_DIR, "test"),
        "csv_path": os.path.join(BASE_CSV_PATH, "PHOENIX-2014-T.test.corpus.csv"),
        "output": "text_to_video_test.jsonl",
    },
}

# Corrupted test files to skip
CORRUPTED_FILES = {
    "20June_2011_Monday_tagesschau-1249",
    "30December_2010_Thursday_tagesschau-2232",
    "30October_2009_Friday_tagesschau-2273",
    "10February_2010_Wednesday_tagesschau-2515",
    "14December_2009_Monday_tagesschau-2092",
    "25August_2010_Wednesday_tagesschau-8516",
    "27August_2009_Thursday_tagesschau-3269",
    "06February_2010_Saturday_tagesschau-7365",
}


# === STEP 1: LOAD TEXT MAP ===
def load_text_map(csv_path):
    text_map = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            name = row["name"]
            sentence = row["translation"].strip()
            text_map[name] = sentence
    return text_map


# === STEP 2: BUILD DATASET ===
def build_dataset(pt_dir, text_map, split):
    dataset = []
    for filename in tqdm(os.listdir(pt_dir)):
        if not filename.endswith(".pt"):
            continue

        name = filename.replace(".pt", "")

        if name not in text_map:
            print(f"[Warning] Missing text for {name}, skipping.")
            continue

        if split == "test" and name in CORRUPTED_FILES:
            print(f"[Skipping corrupted] {name}")
            continue

        pt_path = os.path.join(pt_dir, filename)
        try:
            data = torch.load(pt_path)
            if isinstance(data, tuple):
                data = data[0]
            indices = data.flatten().tolist()
        except Exception as e:
            print(f"[Error] Failed loading {filename}: {e}")
            continue

        dataset.append(
            {
                "video": name,  # 👈 NEW field
                "translation": {"en": text_map[name], "de": " ".join(map(str, indices))},
            }
        )

    return dataset


# === STEP 3: SAVE TO JSONL ===
def save_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


# === MAIN ===
if __name__ == "__main__":
    for split, config in SPLITS.items():
        print(f"\n🔄 Processing {split} split...")
        text_map = load_text_map(config["csv_path"])
        dataset = build_dataset(config["pt_dir"], text_map, split)
        save_jsonl(dataset, config["output"])
        print(f"✅ Saved {len(dataset)} entries to {config['output']}")
