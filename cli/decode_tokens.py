import os
import cv2
import torch
import mediapy as media
import numpy as np
from cosmos_predict1.tokenizer.inference.utils import load_decoder_model

def decode_tokens(token_path, checkpoint_dec, output_path, device="cuda"):
    assert os.path.exists(token_path), f"Token file not found: {token_path}"
    tokens = torch.load(token_path, map_location="cpu")
    print(f"Loaded tokens: {token_path}, shape={tokens.shape}")

    decoder = load_decoder_model(checkpoint_dec, device=device)
    decoder.eval()

    with torch.no_grad():
        tokens = tokens.to(device)
        video = decoder(tokens)

    video = video.detach().cpu().float().numpy()
    if video.ndim == 5:
        video = video[0].transpose(1, 2, 3, 0)
    elif video.ndim == 4:
        video = video.transpose(1, 2, 3, 0)
    else:
        raise ValueError(f"Unexpected shape: {video.shape}")

    print(f"Decoded video shape: {video.shape}, dtype={video.dtype}")

    fps = 25
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        media.write_video(output_path, video, fps=fps, codec="libx264")
        print(f"\u2705 Saved decoded video to {output_path}")
    except Exception as e:
        print(f"[WARN] mediapy failed: {e}, falling back to OpenCV...")
        h, w, _ = video[0].shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in video:
            bgr = cv2.cvtColor((np.clip((frame + 1) * 127.5, 0, 255)).astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(bgr)
        out.release()
        print(f"\u2705 Saved fallback decoded video to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Decode COSMOS video tokens to MP4 video")
    parser.add_argument("--token_path", required=True, help="Path to .pt token file")
    parser.add_argument("--checkpoint_dec", required=True, help="Path to COSMOS decoder .jit file")
    parser.add_argument("--output_path", required=True, help="Output .mp4 video path")
    args = parser.parse_args()

    decode_tokens(
        token_path=args.token_path,
        checkpoint_dec=args.checkpoint_dec,
        output_path=args.output_path,
    )
