# import os
# import torch
# import json
# import csv
# from tqdm import tqdm
#
# # === CONFIGURATION ===
# PT_DIR = os.path.expanduser("~/data/thesis_storage/tokens/train")
# CSV_PATH = os.path.expanduser("~/scratch/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv")
# OUTPUT_JSONL = "video_to_gloss_dataset.jsonl"
#
# # === STEP 1: LOAD CSV (Gloss Annotations) ===
# def load_gloss_map(csv_path):
#     gloss_map = {}
#     with open(csv_path, 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f, delimiter='|')
#         for row in reader:
#             name = row['name']
#             gloss_seq = row['orth'].split()
#             gloss_map[name] = gloss_seq
#     return gloss_map
#
# # === STEP 2: PROCESS .pt FILES AND MATCH GLOSSES ===
# def build_dataset(pt_dir, gloss_map):
#     dataset = []
#     for filename in tqdm(os.listdir(pt_dir)):
#         if not filename.endswith('.pt'):
#             continue
#
#         video_name = filename.replace('.pt', '')
#         if video_name not in gloss_map:
#             print(f"[Warning] Gloss not found for {video_name}, skipping.")
#             continue
#
#         pt_path = os.path.join(pt_dir, filename)
#         data = torch.load(pt_path)
#         codes = data[1]  # index 1 = continuous codes
#         # Shape: [1, 6, T, 8, 8] → flatten to [T, features]
#         codes = codes.squeeze(0)  # → [6, T, 8, 8]
#         T = codes.shape[1]
#         flattened = codes.permute(1, 0, 2, 3).reshape(T, -1)  # → [T, 6*8*8]
#         input_codes = flattened.tolist()
#
#         dataset.append({
#             "input_codes": input_codes,
#             "target_glosses": gloss_map[video_name]
#         })
#
#     return dataset
#
# # === STEP 3: SAVE TO JSONL ===
# def save_jsonl(data, output_path):
#     with open(output_path, 'w', encoding='utf-8') as f:
#         for entry in data:
#             json.dump(entry, f)
#             f.write('\n')
#
# # === MAIN ===
# if __name__ == "__main__":
#     print("Loading gloss annotations...")
#     gloss_map = load_gloss_map(CSV_PATH)
#
#     print("Building dataset from .pt files...")
#     dataset = build_dataset(PT_DIR, gloss_map)
#
#     print(f"Saving {len(dataset)} samples to {OUTPUT_JSONL}...")
#     save_jsonl(dataset, OUTPUT_JSONL)
#
#     print("✅ Done.")

import os
import torch
import json
import csv
import random
from tqdm import tqdm

# === CONFIGURATION ===
PT_DIR = os.path.expanduser("~/data/thesis_storage/tokens/train")
CSV_PATH = os.path.expanduser("~/scratch/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv")
OUTPUT_JSONL = "sample_video_to_gloss_dataset.jsonl"
NUM_SAMPLES = 10

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
def build_sample_dataset(pt_dir, gloss_map, sample_size):
    all_pt_files = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
    random.shuffle(all_pt_files)

    dataset = []
    for filename in tqdm(all_pt_files):
        if len(dataset) >= sample_size:
            break

        video_name = filename.replace('.pt', '')
        if video_name not in gloss_map:
            continue

        pt_path = os.path.join(pt_dir, filename)
        data = torch.load(pt_path)
        codes = data[1]  # index 1 = continuous codes

        # Shape: [1, 6, T, 8, 8] → flatten to [T, 6*8*8]
        codes = codes.squeeze(0)  # → [6, T, 8, 8]
        T = codes.shape[1]
        flattened = codes.permute(1, 0, 2, 3).reshape(T, -1)  # → [T, 384]
        input_codes = flattened.tolist()

        dataset.append({
            "input_codes": input_codes,
            "target_glosses": gloss_map[video_name]
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
    print("Loading gloss annotations...")
    gloss_map = load_gloss_map(CSV_PATH)

    print(f"Building sample dataset of {NUM_SAMPLES} files...")
    dataset = build_sample_dataset(PT_DIR, gloss_map, NUM_SAMPLES)

    print(f"Saving {len(dataset)} samples to {OUTPUT_JSONL}...")
    save_jsonl(dataset, OUTPUT_JSONL)

    print("✅ Sample dataset built and saved.")
