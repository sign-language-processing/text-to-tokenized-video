import sys
from pathlib import Path

# Ensure the cosmos_predict1 directory is on the Python path
project_root = Path(__file__).resolve().parents[2]  # Go two levels up
sys.path.append(str(project_root / "cosmos-predict1"))

from pathlib import Path

import mediapy
import torch
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer


def load_tokenizer(checkpoint_enc: Path = None, checkpoint_dec: Path = None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return CausalVideoTokenizer(
        checkpoint_enc=checkpoint_enc,
        checkpoint_dec=checkpoint_dec,
        device=device,
    )


def decode_tokens(tokens_path: Path, checkpoint_dec: Path, checkpoint_enc: Path = None, device=None):
    data = torch.load(tokens_path)
    # Determine whether data is codes or (indices, codes) depending on encoding mode:
    if isinstance(data, (tuple, list)) and len(data) >= 2:
        latent = data[0]
    elif isinstance(data, torch.Tensor):
        latent = data
    else:
        raise RuntimeError(f"Unexpected tokens file structure: {type(data)}")

    tokenizer = load_tokenizer(checkpoint_enc=checkpoint_enc, checkpoint_dec=checkpoint_dec, device=device)
    video = tokenizer.decode(latent)  # add batch dim if needed
    # video shape: [1, T, H, W, C] or [T, H, W, C] depending on API
    return video.permute(0, 2, 3, 4, 1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=Path, required=True, help="Path to .pt latent file")
    parser.add_argument(
        "--checkpoint-dec", type=Path, required=True, help="Path to decoder checkpoint (e.g. decoder.jit)"
    )
    parser.add_argument("--checkpoint-enc", type=Path, default=None, help="Encoder checkpoint, if needed")
    parser.add_argument(
        "--output-video", type=Path, required=True, help="Path where to save reconstructed video (e.g. out.mp4)"
    )
    parser.add_argument("--fps", type=int, default=25, help="Frame rate to save video")
    args = parser.parse_args()

    video = decode_tokens(
        tokens_path=args.tokens,
        checkpoint_dec=args.checkpoint_dec,
        checkpoint_enc=args.checkpoint_enc,
    )
    # If video has batch dimension
    if video.ndim == 5:
        video = video[0]  # drop batch

    # video: shape [T, H, W, C], values in [-1, 1] or [0,255] depending on model
    # Optionally convert/scale
    video = video.to(dtype=torch.float32).cpu().numpy()
    mediapy.write_video(str(args.output_video), video, fps=args.fps)
    print("Saved reconstructed video to", args.output_video)
