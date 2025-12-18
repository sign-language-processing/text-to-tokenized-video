import os
import torch
import json
import csv
from tqdm import tqdm

# === CONFIGURATION ===
BASE_PT_DIR = os.path.expanduser("~/data/thesis_storage/tokens")
BASE_CSV_PATH = os.path.expanduser("~/scratch/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual")

SPLITS = {
    "train": {
        "pt_dir": os.path.join(BASE_PT_DIR, "train"),
        "csv_path": os.path.join(BASE_CSV_PATH, "PHOENIX-2014-T.train.corpus.csv"),
        "output": "video_to_gloss_train.jsonl"
    },
    "dev": {
        "pt_dir": os.path.join(BASE_PT_DIR, "dev"),
        "csv_path": os.path.join(BASE_CSV_PATH, "PHOENIX-2014-T.dev.corpus.csv"),
        "output": "video_to_gloss_dev.jsonl"
    },
    "test": {
        "pt_dir": os.path.join(BASE_PT_DIR, "test"),
        "csv_path": os.path.join(BASE_CSV_PATH, "PHOENIX-2014-T.test.corpus.csv"),
        "output": "video_to_gloss_test.jsonl"
    },
}

# === STEP 1: LOAD CSV (Gloss Annotations) ===
def load_gloss_map(csv_path):
    gloss_map = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            name = row['name']
            gloss_seq = row['orth'].split()
            gloss_map[name] = gloss_seq
    return gloss_map

# === STEP 2: PROCESS .pt FILES AND MATCH GLOSSES ===
def build_dataset(pt_dir, gloss_map):
    dataset = []
    for filename in tqdm(os.listdir(pt_dir)):
        if not filename.endswith('.pt'):
            continue
        video_name = filename.replace('.pt', '')
        if video_name not in gloss_map:
            print(f"[Warning] Gloss not found for {video_name}, skipping.")
            continue
        pt_path = os.path.join(pt_dir, filename)
        data = torch.load(pt_path)
        indices = data[0].flatten().tolist()
        dataset.append({
            "translations": {
                "en": " ".join(map(str, indices)),
                "de": " ".join(gloss_map[video_name])
            }
        })
    return dataset

# === STEP 3: SAVE TO JSONL ===
def save_jsonl(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

# === MAIN ===
if __name__ == "__main__":
    for split, config in SPLITS.items():
        print(f"\n📂 Processing {split} split...")
        gloss_map = load_gloss_map(config["csv_path"])
        dataset = build_dataset(config["pt_dir"], gloss_map)
        save_jsonl(dataset, config["output"])
        print(f"✅ Saved {len(dataset)} entries to {config['output']}")
