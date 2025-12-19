"""
Cosmos Video Tokenizer CLI

A unified interface for encoding videos to discrete tokens and decoding tokens back to videos
using NVIDIA's Cosmos-1.0-Tokenizer-DV8x16x16.

Based on the Cosmos-Predict1 inference guide:
https://github.com/NVIDIA/Cosmos-Tokenizer
"""

import sys
from functools import cache
from pathlib import Path

# Ensure the cosmos_predict1 directory is on the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "cosmos-predict1"))

import mediapy
import numpy as np
import torch
import torch.nn.functional as F
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer

# =============================================================================
# Tokenizer Loading
# =============================================================================


@cache
def load_tokenizer(
    checkpoint_enc: Path = None, checkpoint_dec: Path = None, device: str = None
) -> CausalVideoTokenizer:
    """
    Load the Cosmos CausalVideoTokenizer with encoder and/or decoder checkpoints.

    Args:
        checkpoint_enc: Path to encoder checkpoint (encoder.jit)
        checkpoint_dec: Path to decoder checkpoint (decoder.jit)
        device: Device to load model on ('cuda' or 'cpu'). Auto-detected if None.

    Returns:
        CausalVideoTokenizer instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return CausalVideoTokenizer(
        checkpoint_enc=checkpoint_enc,
        checkpoint_dec=checkpoint_dec,
        device=device,
    )


# =============================================================================
# Video I/O Utilities
# =============================================================================


def read_video(path: Path) -> np.ndarray:
    """
    Read video frames from a file or directory of PNG images.

    Args:
        path: Path to video file or directory containing PNG frames

    Returns:
        Video array with shape (T, H, W, 3) in RGB format
    """
    if path.is_dir():
        images = sorted(path.glob("*.png"))
        if not images:
            raise ValueError(f"No PNG files found in directory: {path}")
        frames = [mediapy.read_image(str(f)) for f in images]
        return np.stack(frames, axis=0)

    return mediapy.read_video(str(path))


def resize_video(video: torch.Tensor, target_height: int = 128, target_width: int = 128) -> torch.Tensor:
    """
    Resize video to target resolution, ensuring dimensions are divisible by 16.

    Args:
        video: Video tensor with shape (B, T, H, W, C)
        target_height: Target height (will be rounded to multiple of 16)
        target_width: Target width (will be rounded to multiple of 16)

    Returns:
        Resized video tensor with shape (B, T, target_height, target_width, C)
    """
    B, T, H, W, C = video.shape

    # Round to nearest multiple of 16 (required by Cosmos tokenizer)
    target_height = (target_height // 16) * 16
    target_width = (target_width // 16) * 16

    # Reshape for interpolation: (B*T, C, H, W)
    video = video.view(B * T, H, W, C).permute(0, 3, 1, 2)
    video = F.interpolate(video, size=(target_height, target_width), mode="bilinear", align_corners=False)

    # Reshape back: (B, T, H, W, C)
    return video.permute(0, 2, 3, 1).view(B, T, target_height, target_width, C)


# =============================================================================
# Encoding: Video -> Discrete Tokens
# =============================================================================


@torch.no_grad()
def encode(
    video_path: Path, checkpoint_enc: Path, device: str = None, target_height: int = 128, target_width: int = 128
) -> tuple:
    """
    Encode a video file to discrete FSQ tokens.

    Args:
        video_path: Path to video file or directory of PNG frames
        checkpoint_enc: Path to encoder checkpoint (encoder.jit)
        device: Device to use ('cuda' or 'cpu')
        target_height: Target height for resizing
        target_width: Target width for resizing

    Returns:
        Tuple of (indices, codes) from the discrete tokenizer
        - indices: Token indices with shape (B, T', H', W')
        - codes: FSQ codes with shape (B, num_levels, T', H', W')
    """
    # Read video and ensure RGB (drop alpha if present)
    video_np = read_video(video_path)[..., :3]  # (T, H, W, 3)
    video_tensor = torch.from_numpy(video_np)

    # Add batch dimension: (1, T, H, W, 3)
    video_tensor = video_tensor.unsqueeze(0)

    # Resize to target resolution
    video_tensor = resize_video(video_tensor, target_height, target_width)

    # Normalize to [-1, 1] range expected by tokenizer
    video_tensor = video_tensor.to(torch.float32) / 127.5 - 1

    # Rearrange to (B, C, T, H, W) layout expected by tokenizer
    video_tensor = video_tensor.permute(0, 4, 1, 2, 3)

    tokenizer = load_tokenizer(checkpoint_enc=checkpoint_enc, device=device)
    return tokenizer.encode(video_tensor)


# =============================================================================
# Decoding: Discrete Tokens -> Video
# =============================================================================


@torch.no_grad()
def decode(tokens_path: Path, checkpoint_dec: Path, checkpoint_enc: Path = None, device: str = None) -> np.ndarray:
    """
    Decode discrete tokens back to video frames.

    Args:
        tokens_path: Path to .pt file containing saved tokens
        checkpoint_dec: Path to decoder checkpoint (decoder.jit)
        checkpoint_enc: Optional encoder checkpoint (some models require both)
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Video array with shape (T, H, W, C), values in [0, 255] range
    """
    data = torch.load(tokens_path, weights_only=False)

    # Handle different token file formats:
    # - Tuple/list: (indices, codes) from discrete encoding
    # - Tensor: direct indices or latent
    if isinstance(data, (tuple, list)) and len(data) >= 2:
        indices = data[0]
    elif isinstance(data, torch.Tensor):
        indices = data
    else:
        raise RuntimeError(f"Unexpected tokens file structure: {type(data)}")

    tokenizer = load_tokenizer(checkpoint_enc=checkpoint_enc, checkpoint_dec=checkpoint_dec, device=device)

    # Decode tokens to video
    video = tokenizer.decode(indices)

    # Rearrange from (B, C, T, H, W) to (B, T, H, W, C)
    video = video.permute(0, 2, 3, 4, 1)

    # Remove batch dimension if present
    if video.ndim == 5:
        video = video[0]

    # Convert to numpy array
    return video.to(dtype=torch.float32).cpu().numpy()


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cosmos Video Tokenizer - Encode videos to tokens or decode tokens to videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Encode a video:
    python cosmos_tokenizer.py encode --video input.mp4 --checkpoint-enc encoder.jit --output tokens.pt

  Decode tokens:
    python cosmos_tokenizer.py decode --tokens tokens.pt --checkpoint-dec decoder.jit --output video.mp4
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Encode subcommand
    encode_parser = subparsers.add_parser("encode", help="Encode video to discrete tokens")
    encode_parser.add_argument(
        "--video", type=Path, required=True, help="Path to video file or directory of PNG frames"
    )
    encode_parser.add_argument(
        "--checkpoint-enc", type=Path, required=True, help="Path to encoder checkpoint (encoder.jit)"
    )
    encode_parser.add_argument("--output", type=Path, required=True, help="Path to save token output (.pt file)")
    encode_parser.add_argument("--height", type=int, default=128, help="Target height for resizing (default: 128)")
    encode_parser.add_argument("--width", type=int, default=128, help="Target width for resizing (default: 128)")
    encode_parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu, auto-detected if not specified)"
    )

    # Decode subcommand
    decode_parser = subparsers.add_parser("decode", help="Decode tokens back to video")
    decode_parser.add_argument("--tokens", type=Path, required=True, help="Path to .pt token file")
    decode_parser.add_argument(
        "--checkpoint-dec", type=Path, required=True, help="Path to decoder checkpoint (decoder.jit)"
    )
    decode_parser.add_argument(
        "--checkpoint-enc", type=Path, default=None, help="Path to encoder checkpoint (if required)"
    )
    decode_parser.add_argument("--output", type=Path, required=True, help="Path to save reconstructed video (.mp4)")
    decode_parser.add_argument("--fps", type=int, default=25, help="Frame rate for output video (default: 25)")
    decode_parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu, auto-detected if not specified)"
    )

    args = parser.parse_args()

    if args.command == "encode":
        tokens = encode(
            video_path=args.video,
            checkpoint_enc=args.checkpoint_enc,
            device=args.device,
            target_height=args.height,
            target_width=args.width,
        )
        torch.save(tokens, args.output)
        print(f"Saved tokens to {args.output}")
        if isinstance(tokens, tuple):
            print(f"  Indices shape: {tokens[0].shape}")
            print(f"  Codes shape: {tokens[1].shape}")
        else:
            print(f"  Shape: {tokens.shape}")

    elif args.command == "decode":
        video = decode(
            tokens_path=args.tokens,
            checkpoint_dec=args.checkpoint_dec,
            checkpoint_enc=args.checkpoint_enc,
            device=args.device,
        )
        mediapy.write_video(str(args.output), video, fps=args.fps)
        print(f"Saved reconstructed video to {args.output}")
        print(f"  Shape: {video.shape}")


if __name__ == "__main__":
    main()
