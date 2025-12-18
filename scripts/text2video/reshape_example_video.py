#!/usr/bin/env python3

import torch
from pathlib import Path
import sys

# Add cosmos_predict1 to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root / "cosmos-predict1"))

from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer
import mediapy as media

# === CONFIG ===
PRED_PATH = "/home/mpanag/scratch/thesis_storage/text2video_t5small_100epoch/predictions_named/20September_2010_Monday_heute-2939.pt"
OUT_VIDEO = "reconstructed_no_trim.mp4"
DECODER_CKPT = "/home/mpanag/thesis/text-to-tokenized-video/pretrained_ckpts/Cosmos-1.0-Tokenizer-DV8x16x16/decoder.jit"
DEVICE = "cuda"

# === SHAPE ===
T, H, W = 4, 8, 8  # Expected shape
EXPECTED_NUM_TOKENS = T * H * W

# === LOAD FULL PREDICTION ===
flat = torch.load(PRED_PATH).to(torch.long)  # No trimming
print(f"Loaded predicted tokens: shape = {flat.shape}")

# === TRY TO RESHAPE FULL PREDICTION ===
try:
    latent = flat.view(1, T, H, W).to(DEVICE)
    print("✅ Reshape succeeded with full sequence.")
except RuntimeError as e:
    print(f"❌ Reshape failed: {e}")
    print("🧪 Trying automatic trim...")

    # Try trimming automatically to match expected token count
    latent = flat[:EXPECTED_NUM_TOKENS].view(1, T, H, W).to(DEVICE)
    print("✅ Reshape after trim succeeded.")

# === INIT TOKENIZER ===
tokenizer = CausalVideoTokenizer(
    checkpoint_dec=DECODER_CKPT,
    device=DEVICE,
)

# === DECODE ===
with torch.no_grad():
    video = tokenizer.decode(latent)

# === FORMAT & SAVE VIDEO ===
video = video[0].permute(1, 2, 3, 0).float().cpu().numpy()
media.write_video(OUT_VIDEO, video, fps=25)

print(f"🎥 Saved video to: {OUT_VIDEO}")
