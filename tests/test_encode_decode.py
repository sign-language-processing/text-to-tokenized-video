# /text-to-tokenized-video/tests/test_encode_decode.py
# Test case to check the full roundtrip: PNG frames -> Tokens -> Video reconstruction.
# Confirms the core logic, including resizing and padding, by asserting the final decoded
# frame count matches the padded original frame count.

import os
import glob
import math
import cv2
import numpy as np
import torch
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer

# --- CONFIG ---
CHUNK_SIZE = 32
TARGET_SIZE = (128, 128)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Cosmos-1.0-Tokenizer-DV8x16x16"


# --- UTILITY FUNCTIONS (Mirroring encode_dataset.py for self-contained testing) ---

def read_video_glob(frame_glob: str) -> np.ndarray:
    """Reads and stacks PNG frames into a single numpy array (T, H, W, 3)."""
    frame_paths = sorted(glob.glob(frame_glob))
    if not frame_paths:
        raise AssertionError(f"[ERROR] No PNG frames found in: {frame_glob}")
    # Read BGR and convert to RGB
    frames = [cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB) for fp in frame_paths]
    return np.stack(frames, axis=0)  # (T,H,W,3)


def preprocess_video(frames: np.ndarray, target_size=TARGET_SIZE) -> torch.Tensor:
    """Resizes, normalizes, and reshapes video for tokenizer input (1, 3, T, H, W)."""
    resized = np.stack([
        cv2.resize(f, target_size, interpolation=cv2.INTER_AREA)
        for f in frames
    ], axis=0)
    video = torch.from_numpy(resized).float() / 127.5 - 1.0
    video = video.permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,T,H,W)
    return video


def chunk_video(video_torch: torch.Tensor, chunk_size=CHUNK_SIZE, device=DEVICE):
    """Yields video chunks, padding the last chunk if needed."""
    B, C, T, H, W = video_torch.shape
    for start in range(0, T, chunk_size):
        chunk = video_torch[:, :, start:start + chunk_size, :, :]
        if chunk.shape[2] < chunk_size:
            pad_t = chunk_size - chunk.shape[2]
            # Padding with zeros, moved to the correct device
            pad = torch.zeros((B, C, pad_t, H, W), dtype=chunk.dtype, device=device)
            chunk = torch.cat([chunk, pad], dim=2)
        yield chunk


# --- PYTEST FUNCTION ---

def test_encode_decode_roundtrip_frame_count(tmp_path):
    """Tests that a video can be encoded and decoded without frame loss/error."""

    # NOTE: Adjust path to ensure this test sample exists on the cluster
    frame_dir = "/scratch/mpanag/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test/01April_2010_Thursday_heute-6704"
    ckpt_dir = f"checkpoints/{MODEL_NAME}"
    ckpt_enc = f"{ckpt_dir}/encoder.jit"
    ckpt_dec = f"{ckpt_dir}/decoder.jit"

    # --- Setup Checks ---
    if not all(os.path.exists(p) for p in [ckpt_enc, ckpt_dec]):
        # This will only happen if checkpoints aren't downloaded/linked
        from huggingface_hub import snapshot_download
        print(f"Downloading {MODEL_NAME} for test...")
        snapshot_download(repo_id=f"nvidia/{MODEL_NAME}", local_dir=ckpt_dir)

    # 1. Load Original Frames (T_orig)
    video_np = read_video_glob(os.path.join(frame_dir, "*.png"))
    T_orig, _, _, _ = video_np.shape

    # 2. Preprocess, Resize, and Initialize Tokenizer
    video = preprocess_video(video_np)

    tokenizer = CausalVideoTokenizer(
        checkpoint_enc=ckpt_enc,
        checkpoint_dec=ckpt_dec,
        device=DEVICE,
        dtype="bfloat16"
    )

    all_tokens = []

    # 3. Encode in Chunks (Applying Temporal Padding)
    with torch.no_grad():
        for chunk in chunk_video(video, device=DEVICE):
            chunk = chunk.to(device=DEVICE, dtype=torch.bfloat16)
            out = tokenizer.encode(chunk)
            z_indices = out[0] if isinstance(out, tuple) else out
            all_tokens.append(z_indices.cpu())

    tokens = torch.cat(all_tokens, dim=2).int().to(DEVICE)

    # 4. Decode
    decoded = tokenizer.decode(tokens)

    # Decoded shape: (B, 3, T_decoded, H_decoded, W_decoded)
    T_decoded = decoded.shape[2]

    # Calculate the total number of frames that *were actually encoded* (due to padding)
    T_encoded_padded = math.ceil(T_orig / CHUNK_SIZE) * CHUNK_SIZE

    # 5. Assertion (Checking decoded length matches padded encoded length)
    assert T_decoded == T_encoded_padded, \
        (f"Mismatch in frame count. Original Frames: {T_orig}, "
         f"Expected Decoded Frames (Padded): {T_encoded_padded}, "
         f"Actual Decoded Frames: {T_decoded}")

    print(f"✅ Encode/decode roundtrip test passed successfully.")
    print(f"   Original Frames: {T_orig}, Decoded Frames: {T_decoded} (Padded)")
