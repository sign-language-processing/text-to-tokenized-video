#!/usr/bin/env python3

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from jiwer import wer


# ============================================================
# CORRECT DATASET (MATCHES TRAINING)
# ============================================================

class VideoLatents2GlossDataset(Dataset):
    def __init__(self, csv_path, pt_dir, vocab, max_video_len=256, target_channels=60):
        self.df = pd.read_csv(csv_path, sep="|")
        self.pt_dir = pt_dir
        self.max_video_len = max_video_len
        self.vocab = vocab
        self.id2gloss = {i: g for g, i in vocab.items()}
        self.target_channels = target_channels

        # Only keep rows with existing .pt files
        self.df = self.df[self.df["name"].apply(
            lambda n: os.path.isfile(os.path.join(pt_dir, f"{n}.pt"))
        )].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def pad_channels(self, tensor):
        # tensor shape: [T, H, W, C]
        T, H, W, C = tensor.shape
        if C < self.target_channels:
            pad = torch.zeros((T, H, W, self.target_channels - C), dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad], dim=-1)
        else:
            tensor = tensor[..., :self.target_channels]
        return tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["name"]

        path = os.path.join(self.pt_dir, f"{name}.pt")
        tensor = torch.load(path, map_location="cpu")

        if isinstance(tensor, tuple):
            tensor = tensor[0]

        tensor = tensor.squeeze(0)  # [T, H, W, C]
        tensor = self.pad_channels(tensor)

        T = min(tensor.shape[0], self.max_video_len)
        tensor = tensor[:T]

        # pad time dimension
        if T < self.max_video_len:
            pad = torch.zeros((self.max_video_len - T, *tensor.shape[1:]))
            tensor = torch.cat([tensor, pad], dim=0)

        # flatten to [T, 15360]
        tensor = tensor.reshape(self.max_video_len, -1)

        gloss = row["orth"].split()
        return {
            "video_tokens": tensor.float(),
            "gloss_ref": gloss,
        }


def collate_fn(batch):
    vt = torch.stack([b["video_tokens"] for b in batch], dim=0)
    gloss_refs = [b["gloss_ref"] for b in batch]
    return {"video_tokens": vt, "gloss_ref": gloss_refs}


# ============================================================
# MODEL
# ============================================================

class Seq2SeqVideoToGloss(nn.Module):
    def __init__(self, token_dim, vocab_size, hidden_dim=512, num_layers=4, nhead=8, dropout=0.1):
        super().__init__()
        self.encoder_proj = nn.Linear(token_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self):
        raise NotImplementedError("Use greedy_decode instead.")

    def greedy_decode(self, video_tokens, sos_id, eos_id, max_len=64, device='cpu'):
        B, T, D = video_tokens.shape

        enc = self.encoder_proj(video_tokens)
        memory = self.encoder(enc)

        decoded = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        results = [[] for _ in range(B)]
        finished = [False] * B

        for step in range(max_len):
            tgt_emb = self.embed(decoded)
            dec_out = self.decoder(tgt_emb, memory)
            logits = self.classifier(dec_out)
            next_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_logits, dim=-1)

            decoded = torch.cat([decoded, next_tokens.unsqueeze(1)], dim=1)

            for i, tok in enumerate(next_tokens.tolist()):
                if not finished[i]:
                    if tok == eos_id:
                        finished[i] = True
                    else:
                        results[i].append(tok)

            if all(finished):
                break

        return results


def load_model(model_path, vocab_size, device):
    ckpt = torch.load(model_path, map_location=device)
    sd = ckpt["model_state_dict"]
    token_dim = sd["encoder_proj.weight"].shape[1]
    model = Seq2SeqVideoToGloss(token_dim, vocab_size).to(device)
    model.load_state_dict(sd)
    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(csv_path, pt_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device)
    vocab = ckpt["vocab"]
    id2gloss = {i: g for g, i in vocab.items()}

    # Load dataset
    ds = VideoLatents2GlossDataset(csv_path, pt_dir, vocab)
    print("[DEBUG] Loaded samples:", len(ds))
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Load model
    model = load_model(model_path, len(vocab), device)
    model.eval()

    sos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]
    print("[DEBUG] SOS ID:", sos_id, "EOS ID:", eos_id)
    print("[DEBUG] SOS Gloss:", id2gloss.get(sos_id), "EOS Gloss:", id2gloss.get(eos_id))

    refs, hyps = [], []

    for i, batch in enumerate(dl):
        vt = batch["video_tokens"].to(device)

        with torch.no_grad():
            pred_ids = model.greedy_decode(vt, sos_id, eos_id, device=device)[0]

        print(f"\n[Sample {i}]")
        print("Predicted IDs:", pred_ids)

        hyp_tokens = []
        for tid in pred_ids:
            gloss = id2gloss.get(tid, "<UNK>")
            if gloss == "<UNK>":
                print(f"[WARN] Unknown token ID: {tid}")
            hyp_tokens.append(gloss)

        hyp = " ".join(hyp_tokens)
        ref = " ".join(batch["gloss_ref"][0])

        print("REF:", ref)
        print("HYP:", hyp)

        refs.append(ref)
        hyps.append(hyp)

        if i >= 4:
            break  # only debug 5 examples

    print("\n=== FINAL RESULTS (first 5 samples) ===")
    print(f"WER: {wer(refs, hyps) * 100:.2f}%")


if __name__ == "__main__":
    evaluate(
        csv_path=os.path.expanduser(
            "~/scratch/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv"
        ),
        pt_dir=os.path.expanduser("~/data/thesis_storage/latentfeatures/test"),
        model_path="video2gloss_seq2seq_cosmos.pt",
    )
