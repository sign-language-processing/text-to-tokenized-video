import sys
from pathlib import Path

# Ensure the cosmos_predict1 directory is on the Python path
project_root = Path(__file__).resolve().parents[2]  # Go two levels up
sys.path.append(str(project_root / "cosmos-predict1"))

from functools import cache
from pathlib import Path

import mediapy
import numpy as np
import torch
import torch.nn.functional as F
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer


def read_video(directory_or_file: Path):
    """Reads image frames in BGR, converts to RGB, returns shape (1, T, H, W, 3)"""
    if directory_or_file.is_dir():
        images = directory_or_file.glob("*.png")
        frames = [mediapy.read_image(f) for f in images]
        return np.stack(frames, axis=0)

    return mediapy.read_video(directory_or_file)


@cache
def load_tokenizer(checkpoint_enc: Path = None, checkpoint_dec: Path = None, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    return CausalVideoTokenizer(
        checkpoint_enc=checkpoint_enc,
        checkpoint_dec=checkpoint_dec,
        device=device,
    )


def resize_videos(videos, target_height=128, target_width=128):
    """Resize videos to target resolution, ensuring dimensions are divisible by 16."""
    B, T, H, W, C = videos.shape

    # Round to nearest multiple of 16
    target_height = (target_height // 16) * 16
    target_width = (target_width // 16) * 16

    videos = videos.view(B * T, H, W, C).permute(0, 3, 1, 2)  # [B*T, C, H, W]
    videos = F.interpolate(videos, size=(target_height, target_width), mode="bilinear", align_corners=False)
    return videos.permute(0, 2, 3, 1).view(B, T, target_height, target_width, C)


@torch.no_grad()
def encode_video(video, checkpoint_enc, device=None):
    video_np = read_video(video)[..., :3]  # (T, H, W, 3)
    video_tensor = torch.from_numpy(video_np)

    # Add batch dimension: (1, T, H, W, 3)
    video_tensor = video_tensor.unsqueeze(0)
    video_tensor = resize_videos(video_tensor)

    # Normalize tensor
    video_tensor = video_tensor.to(torch.float) / 127.5 - 1

    # Rearrange to Bx3xTxHxW layout expected by tokenizer
    video_tensor = video_tensor.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)

    tokenizer = load_tokenizer(checkpoint_enc=checkpoint_enc, device=device)
    return tokenizer.encode(video_tensor)  # returns tokens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint-enc", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    tokens = encode_video(
        video=args.video,
        checkpoint_enc=args.checkpoint_enc,
    )

    torch.save(tokens, args.output_path)
    print(f"Saved tokens to {args.output_path}")
    print(tokens)
