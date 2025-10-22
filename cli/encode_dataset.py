# -*- coding: utf-8 -*-
"""
GPU-safe tokenization of PHOENIX-2014-T using NVIDIA Cosmos Discrete Tokenizer (DV8x16x16).
Fully fixed version:
- Uses BF16 (A100)
- Resizes to 128x128
- Pads/truncates token chunks for consistent temporal dimension
- Avoids skipping videos due to token shape mismatches
"""

import os
import glob
import time
import cv2
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer

# -----------------------------
# Precision and CUDA settings
# -----------------------------
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# -----------------------------
# Config
# -----------------------------
BASE_FRAMES = "/scratch/mpanag/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"
ANNOTATIONS = "/scratch/mpanag/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual"
OUT_DIR = "/data/mpanag/thesis_storage/tokens"
MODEL_NAME = "Cosmos-1.0-Tokenizer-DV8x16x16"
CHUNK_SIZE = 32
TARGET_SIZE = (128, 128)
DEVICE = "cuda"

os.makedirs(OUT_DIR, exist_ok=True)
ckpt_dir = f"checkpoints/{MODEL_NAME}"

if not os.path.exists(ckpt_dir):
    print(f"Downloading {MODEL_NAME} from Hugging Face...")
    snapshot_download(repo_id=f"nvidia/{MODEL_NAME}", local_dir=ckpt_dir)

# -----------------------------
# Utils
# -----------------------------
def read_video(frame_glob: str) -> np.ndarray:
    frame_paths = sorted(glob.glob(frame_glob))
    if not frame_paths:
        raise ValueError(f"No frames found: {frame_glob}")
    frames = [cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB) for fp in frame_paths]
    return np.stack(frames, axis=0)

def preprocess_video(frames: np.ndarray, target_size=(128, 128)) -> torch.Tensor:
    resized = np.stack([
        cv2.resize(f, target_size, interpolation=cv2.INTER_AREA)
        for f in frames
    ], axis=0)
    video = torch.from_numpy(resized).float() / 127.5 - 1.0
    video = video.permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,T,H,W)
    return video

def chunk_video(video_torch: torch.Tensor, chunk_size=32):
    B, C, T, H, W = video_torch.shape
    for start in range(0, T, chunk_size):
        chunk = video_torch[:, :, start:start + chunk_size, :, :]
        if chunk.shape[2] < chunk_size:
            pad_t = chunk_size - chunk.shape[2]
            pad = torch.zeros((B, C, pad_t, H, W), dtype=chunk.dtype, device=chunk.device)
            chunk = torch.cat([chunk, pad], dim=2)
        yield chunk

# -----------------------------
# Main process
# -----------------------------
def process_split(split: str, corpus_file: str, limit: int = None):
    df = pd.read_csv(corpus_file, sep="|")
    if limit:
        df = df.head(limit)
    print(f"Processing {split}: {len(df)} sequences")

    results = []

    tokenizer = CausalVideoTokenizer(
        checkpoint_enc=f"{ckpt_dir}/encoder.jit",
        device=DEVICE,
        dtype="bfloat16",
    )

    if hasattr(tokenizer, "model"):
        for p in tokenizer.model.parameters():
            p.data = p.data.float()

    for _, row in df.iterrows():
        seq_id = row["name"]
        gloss = row.get("orth", None)
        text = row.get("translation", None)
        frame_glob = f"{BASE_FRAMES}/{split}/{seq_id}/*.png"

        try:
            frames = read_video(frame_glob)
        except Exception as e:
            print(f"[WARN] Skipping {seq_id}: {e}")
            continue

        video = preprocess_video(frames, target_size=TARGET_SIZE).to(DEVICE, dtype=torch.float32)
        all_tokens = []
        t0 = time.time()

        with torch.no_grad():
            for i, chunk in enumerate(chunk_video(video, chunk_size=CHUNK_SIZE)):
                try:
                    chunk = chunk.to(device=DEVICE, dtype=torch.bfloat16)
                    out = tokenizer.encode(chunk)
                    z_indices = out[0] if isinstance(out, tuple) else out
                    all_tokens.append(z_indices.cpu())
                    print(f"[DEBUG] {seq_id} - Chunk {i} token shape: {tuple(z_indices.shape)}")
                except RuntimeError as e:
                    print(f"[ERROR] Chunk failed for {seq_id} (chunk {i}): {e}")
                    continue

        if not all_tokens:
            print(f"[WARN] No tokens generated for {seq_id}")
            continue

        # 🔧 Dynamic temporal padding for consistent shape
        try:
            max_t = max(t.shape[2] for t in all_tokens)
            padded_tokens = []
            for t in all_tokens:
                if t.shape[2] < max_t:
                    pad_t = max_t - t.shape[2]
                    pad = torch.zeros((t.shape[0], t.shape[1], pad_t, t.shape[3], t.shape[4]),
                                      dtype=t.dtype)
                    t = torch.cat([t, pad], dim=2)
                padded_tokens.append(t)
            tokens = torch.cat(padded_tokens, dim=2).int()
        except RuntimeError as e:
            print(f"[WARN] Skipping {seq_id} due to concat error: {e}")
            continue

        runtime_sec = time.time() - t0

        token_outdir = os.path.join(OUT_DIR, split, MODEL_NAME)
        os.makedirs(token_outdir, exist_ok=True)
        token_path = os.path.join(token_outdir, f"{seq_id}.pt")
        torch.save(tokens, token_path)

        results.append({
            "split": split,
            "seq_id": seq_id,
            "gloss": gloss,
            "text": text,
            "model": MODEL_NAME,
            "n_frames": frames.shape[0],
            "height": frames.shape[1],
            "width": frames.shape[2],
            "preprocessing": f"resize({TARGET_SIZE[0]}x{TARGET_SIZE[1]})+chunk{CHUNK_SIZE}+bf16",
            "token_path": token_path,
            "runtime_sec": runtime_sec,
        })

        print(f"✅ {seq_id}: saved tokens {tuple(tokens.shape)} in {runtime_sec:.1f}s")

    out_csv = os.path.join(OUT_DIR, f"tokens_{split}.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"📄 Saved {len(results)} entries to {out_csv}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    splits = {
        "train": f"{ANNOTATIONS}/PHOENIX-2014-T.train.corpus.csv",
        "dev": f"{ANNOTATIONS}/PHOENIX-2014-T.dev.corpus.csv",
        "test": f"{ANNOTATIONS}/PHOENIX-2014-T.test.corpus.csv",
    }

    for split, csv in splits.items():
        process_split(split, csv, limit=args.limit)
