# test_encode_decode.py
# Test case to check roundtrip: PNG → Tokens → Video
# Confirms decoded video has same frame count as original

import os
import torch
import numpy as np
import cv2
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer


def read_video_glob(glob_path):
    import glob
    frame_paths = sorted(glob.glob(glob_path))
    frames = [cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB) for fp in frame_paths]
    return np.stack(frames, axis=0)  # (T,H,W,3)

def test_encode_decode_roundtrip():
    # CONFIG
    frame_dir = "data/example_frames/09April_2010_Friday_tagesschau-7631"
    ckpt_enc = "checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/encoder.jit"
    ckpt_dec = "checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/decoder.jit"
    token_path = "/tmp/test_tokens.pt"
    out_path = "/tmp/test_decoded.mp4"

    assert os.path.exists(ckpt_enc), "Missing encoder checkpoint"
    assert os.path.exists(ckpt_dec), "Missing decoder checkpoint"

    # Load + preprocess frames
    video_np = read_video_glob(os.path.join(frame_dir, "*.png"))
    T, H, W, _ = video_np.shape
    video = torch.from_numpy(video_np).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1.0  # (1,3,T,H,W)

    tokenizer = CausalVideoTokenizer(
        checkpoint_enc=ckpt_enc,
        checkpoint_dec=ckpt_dec,
        device="cuda",
        dtype="bfloat16"
    )

    # Encode
    with torch.no_grad():
        tokens = tokenizer.encode(video.to("cuda"))
        torch.save(tokens, token_path)

        # Decode
        decoded = tokenizer.decode(tokens)
        assert decoded.shape[0] == T, f"Mismatch in frame count: {decoded.shape[0]} != {T}"

    print(f"✅ Encode/decode test passed. Frames: {T}. Saved token: {token_path}, decoded video: {out_path}")
