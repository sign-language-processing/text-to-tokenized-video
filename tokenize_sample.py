# -*- coding: utf-8 -*-
"""
Tokenize PHOENIX PNG frames with NVIDIA Cosmos Tokenizer
Cluster-safe version (no GUI) – runs both CV8x8x8 and DV8x16x16.
Fix: ensures RGB channel order to avoid blue tint issue.
Also logs metrics (PSNR/SSIM) + runtime to documentation/summary/.
"""

import os
import glob
import time
import cv2
import numpy as np
import torch
import mediapy as media
from huggingface_hub import snapshot_download
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer
from documentation.summary.log_results import log_run  # <-- NEW

# ----------------------------------------------------------
# 1) Define which models you want to use
# ----------------------------------------------------------
model_names = [
    "Cosmos-1.0-Tokenizer-CV8x8x8",
    "Cosmos-1.0-Tokenizer-DV8x16x16",
]

# Download checkpoints if missing
for model_name in model_names:
    local_dir = f"checkpoints/{model_name}"
    if not os.path.exists(local_dir):
        print(f"Downloading {model_name}...")
        snapshot_download(repo_id=f"nvidia/{model_name}", local_dir=local_dir)
    else:
        print(f"Found existing checkpoints for {model_name}")

# ----------------------------------------------------------
# 2) Select one PHOENIX sequence as test input
# ----------------------------------------------------------
input_glob = "/scratch/mpanag/PHOENIX-2014-T-128x128/train/09April_2010_Friday_tagesschau-7631/*.png"

def read_video(frame_glob: str) -> np.ndarray:
    """Reads sequence of PNG frames into (T, H, W, 3) **RGB** numpy array"""
    frame_paths = sorted(glob.glob(frame_glob))
    if not frame_paths:
        raise ValueError(f"No frames found matching {frame_glob}")
    frames = [cv2.imread(fp, cv2.IMREAD_COLOR) for fp in frame_paths]
    if any(f is None for f in frames):
        raise ValueError(f"Some frames could not be read in {frame_glob}")
    # Convert BGR (cv2 default) → RGB
    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    return np.stack(frames, axis=0)

def _to_float01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    m = a.max()
    if m > 1.0:
        a /= 255.0
    return np.clip(a, 0.0, 1.0)

def compute_metrics(gt_rgb: np.ndarray, pred_rgb: np.ndarray) -> dict:
    """
    gt_rgb, pred_rgb: (T,H,W,3) RGB arrays, values uint8 or [0,1]/float.
    Returns dict with psnr_mean and ssim_mean (ssim if available).
    """
    T = min(len(gt_rgb), len(pred_rgb))
    gt = _to_float01(gt_rgb[:T])
    pr = _to_float01(pred_rgb[:T])

    # PSNR
    def psnr(a, b, data_range=1.0):
        mse = np.mean((a - b) ** 2)
        if mse == 0:
            return float("inf")
        return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)

    psnrs = [psnr(gt[t], pr[t], 1.0) for t in range(T)]

    # SSIM (optional)
    ssim_vals = []
    try:
        from skimage.metrics import structural_similarity as ssim
        for t in range(T):
            try:
                val = ssim(gt[t], pr[t], channel_axis=2, data_range=1.0)
            except TypeError:
                # for older scikit-image
                val = ssim(gt[t], pr[t], multichannel=True, data_range=1.0)
            ssim_vals.append(val)
        ssim_mean = float(np.mean(ssim_vals))
    except Exception:
        ssim_mean = None

    return {
        "psnr_mean": float(np.mean(psnrs)),
        "ssim_mean": ssim_mean,
    }

# Load input frames
input_video = read_video(input_glob)
print(f"Loaded video: shape={input_video.shape}")  # T x H x W x 3 (RGB)
T, H, W, _ = input_video.shape

batched_input_video = np.expand_dims(input_video, axis=0)  # B x T x H x W x C

# ----------------------------------------------------------
# 3) Run through both models
# ----------------------------------------------------------
os.makedirs("reconstructions", exist_ok=True)
temporal_window = 49
dtype_str = "bfloat16"  # recorded for logs

for model_name in model_names:
    encoder_ckpt = f"checkpoints/{model_name}/encoder.jit"
    decoder_ckpt = f"checkpoints/{model_name}/decoder.jit"

    print(f"\n=== Running model {model_name} ===")
    t0 = time.time()

    tokenizer = CausalVideoTokenizer(
        checkpoint_enc=encoder_ckpt,
        checkpoint_dec=decoder_ckpt,
        device="cuda",
        dtype="bfloat16",  # change to float32 if GPU complains
    )

    batched_output_video = tokenizer(batched_input_video, temporal_window=temporal_window)
    output_video = batched_output_video[0]  # (T,H,W,3), typically RGB in [0,1] or uint8

    # Save reconstruction (mediapy expects RGB)
    base_name = os.path.basename(os.path.dirname(input_glob))
    out_path = f"reconstructions/{base_name}_{model_name}.mp4"
    saved_as = "mp4"

    try:
        media.write_video(out_path, output_video, fps=25)
        print(f"Saved reconstruction: {out_path}")
    except Exception as e:
        print(f"[WARN] mediapy failed ({e}). Falling back to PNG frames...")
        frame_dir = out_path.replace(".mp4", "_frames")
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(output_video):
            media.write_image(os.path.join(frame_dir, f"frame_{i:05d}.png"), frame)
        out_path = frame_dir
        saved_as = "frames"
        print(f"Frames saved in: {frame_dir}")

    runtime_sec = time.time() - t0

    # --- Metrics vs original frames ---
    try:
        metrics = compute_metrics(input_video, output_video)
        print(f"[METRICS] PSNR={metrics['psnr_mean']:.2f} dB"
              + (f", SSIM={metrics['ssim_mean']:.3f}" if metrics['ssim_mean'] is not None else ", SSIM=N/A"))
    except Exception as e:
        metrics = {"psnr_mean": None, "ssim_mean": None}
        print(f"[WARN] metric computation failed: {e}")

    # --- Log the run ---
    log_run(
        model_name=model_name,
        input_glob=input_glob,
        n_frames=T,
        height=H,
        width=W,
        temporal_window=temporal_window,
        dtype=dtype_str,
        runtime_sec=runtime_sec,
        saved_as=saved_as,
        out_path=out_path,
        metrics=metrics,
        notes="Baseline run with RGB-fix",
    )
