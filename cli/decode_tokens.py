
# This script decodes a sequence of discrete video tokens (.pt) into a reconstructed video using the COSMOS Tokenizer decoder.
# It loads tokens from a .pt file, decodes them using the decoder JIT checkpoint, and saves an output video (MP4 or PNG frames).

import os
import torch
import mediapy as media
from cosmos_predict1.tokenizer.inference.utils import load_decoder_model


def decode_tokens(token_path, checkpoint_dec, output_path, device="cuda"):
    # Load token tensor
    assert os.path.exists(token_path), f"Token file not found: {token_path}"
    tokens = torch.load(token_path, map_location="cpu")
    print(f"Loaded tokens: {token_path}, shape={tokens.shape}")

    # Load decoder
    decoder = load_decoder_model(checkpoint_dec, device=device)
    decoder.eval()

    with torch.no_grad():
        tokens = tokens.to(device)
        video = decoder.decode(tokens)

    # Convert to float32 for mediapy compatibility
    video = video.detach().cpu().float().numpy()

    # Rearrange: (B, C, T, H, W) → (T, H, W, C)
    if video.ndim == 5:
        video = video[0].transpose(1, 2, 3, 0)
    elif video.ndim == 4:
        video = video.transpose(1, 2, 3, 0)
    else:
        raise ValueError(f"Unexpected shape: {video.shape}")

    # Write video
    fps = 25
    media.write_video(output_path, video, fps=fps)
    print(f" Saved decoded video to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_path", required=True, help="Path to .pt token file")
    parser.add_argument("--checkpoint_dec", required=True, help="Path to COSMOS decoder .jit file")
    parser.add_argument("--output_path", required=True, help="Output .mp4 video path")
    args = parser.parse_args()

    decode_tokens(
        token_path=args.token_path,
        checkpoint_dec=args.checkpoint_dec,
        output_path=args.output_path,
    )
