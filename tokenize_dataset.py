# -*- coding: utf-8 -*-
"""
Tokenize the full PHOENIX-2014-T dataset with NVIDIA Cosmos Tokenizers.
Saves discrete token IDs (z_indices) for each sequence.
Preprocessing: crops inputs (T,H,W) to multiples of 16.
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
# Config
# -----------------------------
RESOLUTION = "128x128"
BASE_FRAMES = "/scratch/mpanag/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"
ANNOTATIONS = "/scratch/mpanag/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual"
OUT_DIR = "/data/mpanag/thesis_storage/tokens"

os.makedirs(OUT_DIR, exist_ok=True)

model_names = [
    "Cosmos-1.0-Tokenizer-CV8x8x8",
    "Cosmos-1.0-Tokenizer-DV8x16x16",
]

# Ensure checkpoints exist
for model_name in model_names:
    ckpt_dir = f"checkpoints/{model_name}"
    if not os.path.exists(ckpt_dir):
        print(f"Downloading {model_name}...")
        snapshot_download(repo_id=f"nvidia/{model_name}", local_dir=ckpt_dir)

# -----------------------------
# Utils
# -----------------------------
def read_video(frame_glob: str) -> np.ndarray:
    """Reads PNG frames into (T,H,W,3) RGB numpy array."""
    frame_paths = sorted(glob.glob(frame_glob))
    if not frame_paths:
        raise ValueError(f"No frames found: {frame_glob}")
    frames = [cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB) for fp in frame_paths]
    return np.stack(frames, axis=0)

def preprocess_video(frames: np.ndarray) -> torch.Tensor:
    """Convert numpy video (T,H,W,3) → torch tensor (B,3,T,H,W), normalized [-1,1]."""
    video = np.expand_dims(frames, axis=0)           # B,T,H,W,3
    video_torch = torch.from_numpy(video).float()    # float32
    video_torch = video_torch.permute(0, 4, 1, 2, 3) # B,3,T,H,W
    video_torch = (video_torch / 127.5) - 1.0        # scale to [-1,1]
    return video_torch

def crop_to_multiple(x: torch.Tensor, mult: int = 16) -> torch.Tensor:
    """Crop tensor (B,3,T,H,W) so T,H,W are divisible by mult."""
    _, _, T, H, W = x.shape
    T_new = (T // mult) * mult
    H_new = (H // mult) * mult
    W_new = (W // mult) * mult
    return x[:, :, :T_new, :H_new, :W_new]

# -----------------------------
# Main tokenization loop
# -----------------------------
def process_split(split: str, corpus_file: str, limit: int = None):
    df = pd.read_csv(corpus_file, sep="|")
    if limit:
        df = df.head(limit)
    print(f"Processing {split}: {len(df)} sequences")

    results = []

    for _, row in df.iterrows():
        seq_id = row["name"]
        gloss = row.get("orth", None)
        text = row.get("translation", None)

        frame_glob = f"{BASE_FRAMES}/{split}/{seq_id}/*.png"
        try:
            input_video = read_video(frame_glob)
        except Exception as e:
            print(f"[WARN] Skipping {seq_id}: {e}")
            continue

        video_torch = preprocess_video(input_video)
        video_torch = crop_to_multiple(video_torch, mult=16)
        video_torch = video_torch.to("cuda", dtype=torch.bfloat16)

        for model_name in model_names:
            encoder_ckpt = f"checkpoints/{model_name}/encoder.jit"
            decoder_ckpt = f"checkpoints/{model_name}/decoder.jit"

            tokenizer = CausalVideoTokenizer(
                checkpoint_enc=encoder_ckpt,
                checkpoint_dec=decoder_ckpt,
                device="cuda",
                dtype="bfloat16",
            )

            t0 = time.time()
            out = tokenizer.encode(video_torch)

            # Cosmos returns (z_indices, latents). Keep z_indices only.
            if isinstance(out, tuple):
                tokens = out[0]
            else:
                tokens = out

            tokens = tokens.int().cpu()
            runtime_sec = time.time() - t0

            # Save tokens
            token_outdir = os.path.join(OUT_DIR, split, model_name)
            os.makedirs(token_outdir, exist_ok=True)
            token_path = os.path.join(token_outdir, f"{seq_id}.pt")
            torch.save(tokens, token_path)

            results.append({
                "split": split,
                "seq_id": seq_id,
                "gloss": gloss,
                "text": text,
                "model": model_name,
                "n_frames": input_video.shape[0],
                "height": input_video.shape[1],
                "width": input_video.shape[2],
                "preprocessing": "crop_to_multiple(16)",
                "token_path": token_path,
                "runtime_sec": runtime_sec,
            })

    out_csv = os.path.join(OUT_DIR, f"tokens_{split}.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved {len(results)} entries to {out_csv}")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of sequences per split (for testing).")
    args = parser.parse_args()

    splits = {
        "train": f"{ANNOTATIONS}/PHOENIX-2014-T.train.corpus.csv",
        "dev": f"{ANNOTATIONS}/PHOENIX-2014-T.dev.corpus.csv",
        "test": f"{ANNOTATIONS}/PHOENIX-2014-T.test.corpus.csv",
    }
    for split, corpus_file in splits.items():
        process_split(split, corpus_file, limit=args.limit)
