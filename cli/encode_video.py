import os
import glob
import cv2
import torch
import numpy as np
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer

def read_video(frame_dir):
    """Reads image frames in BGR, converts to RGB, returns shape (1, T, H, W, 3)"""
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, '*.png')))
    if not frame_paths:
        raise ValueError(f"No frames found in {frame_dir}")
    frames = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in frame_paths]
    video_np = np.stack(frames, axis=0)
    return video_np  # Shape: (T, H, W, 3)

def encode_video(frame_dir, checkpoint_enc, output_path, device="cuda"):
    video_np = read_video(frame_dir)  # (T, H, W, 3)
    video_np = np.expand_dims(video_np, axis=0)  # (1, T, H, W, 3)
    video_tensor = torch.from_numpy(video_np).to(device).float() / 255.0  # (1, T, H, W, 3)

    tokenizer = CausalVideoTokenizer(
        checkpoint_enc=checkpoint_enc,
        checkpoint_dec=None,
        device=device,
        dtype="bfloat16",  # or "float32" if needed
    )

    with torch.no_grad():
        tokens = tokenizer.encode(video_tensor)  # returns tokens

    torch.save(tokens, output_path)
    print(f"Saved tokens to {output_path}")

# Example usage (keep this under if __name__ == "__main__": if you want)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dir", required=True)
    parser.add_argument("--checkpoint_enc", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    encode_video(
        frame_dir=args.frame_dir,
        checkpoint_enc=args.checkpoint_enc,
        output_path=args.output_path
    )
