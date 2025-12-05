"""
Seq2Seq Transformer for translating tokenized video to gloss sequences.

This script pads all video-token sequences to a consistent shape (T, C, H, W),
normalizes the data, and trains a Transformer-based seq2seq model.
"""

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class VideoToken2GlossDataset(Dataset):
    def __init__(self, csv_path, pt_dir, vocab=None,
                 max_video_len=256, max_gloss_len=64,
                 target_channels=60, target_H=8, target_W=8):
        self.df = pd.read_csv(csv_path, sep="|")
        self.pt_dir = pt_dir
        self.max_video_len = max_video_len
        self.max_gloss_len = max_gloss_len
        self.target_channels = target_channels
        self.target_H = target_H
        self.target_W = target_W

        if vocab is None:
            all_glosses = set()
            for gloss_seq in self.df["orth"]:
                all_glosses.update(gloss_seq.strip().split())
            self.vocab = {g: i + 1 for i, g in enumerate(sorted(all_glosses))}
            self.vocab["<pad>"] = 0
            self.vocab["<bos>"] = len(self.vocab)
            self.vocab["<eos>"] = len(self.vocab)
        else:
            self.vocab = vocab

        self.id2gloss = {i: g for g, i in self.vocab.items()}

        self.df = self.df[self.df["name"].apply(
            lambda n: os.path.isfile(os.path.join(pt_dir, f"{n}.pt"))
        )].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def pad_channels(self, frame: torch.Tensor):
        C, H, W = frame.shape
        if C < self.target_channels:
            pad_c = torch.zeros((self.target_channels - C, H, W), dtype=frame.dtype)
            frame = torch.cat([frame, pad_c], dim=0)
        else:
            frame = frame[:self.target_channels]
        return frame

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["name"]
        token_path = os.path.join(self.pt_dir, f"{name}.pt")
        video_tokens = torch.load(token_path, map_location="cpu")
        if isinstance(video_tokens, tuple):
            video_tokens = video_tokens[0]

        tokens = video_tokens[: self.max_video_len]
        T = tokens.shape[0]

        frames = [self.pad_channels(tokens[i]) for i in range(T)]
        frames = torch.stack(frames, dim=0)

        if T < self.max_video_len:
            pad_frames = torch.zeros((self.max_video_len - T,
                                      self.target_channels,
                                      self.target_H,
                                      self.target_W),
                                     dtype=frames.dtype)
            frames = torch.cat([frames, pad_frames], dim=0)

        gloss = row["orth"].split()
        gloss = gloss[: self.max_gloss_len - 2]
        target = ["<bos>"] + gloss + ["<eos>"]
        target_ids = [self.vocab.get(g, 0) for g in target]
        if len(target_ids) < self.max_gloss_len:
            target_ids += [self.vocab["<pad>"]] * (self.max_gloss_len - len(target_ids))
        else:
            target_ids = target_ids[: self.max_gloss_len]

        return {
            "video_tokens": frames.float(),
            "target_ids": torch.tensor(target_ids, dtype=torch.long)
        }

def collate_fn(batch):
    vt = torch.stack([b["video_tokens"] for b in batch], dim=0)
    tgt = torch.stack([b["target_ids"] for b in batch], dim=0)
    return {"video_tokens": vt, "target_ids": tgt}

class Seq2SeqVideoToGloss(nn.Module):
    def __init__(self, token_dim, vocab_size, hidden_dim=512, num_layers=4, nhead=8, dropout=0.1):
        super().__init__()
        self.encoder_proj = nn.Linear(token_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        dec_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead,
                                               dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, video_tokens, target_ids=None):
        x = video_tokens  # B, T, C*H*W
        enc = self.encoder_proj(x)
        memory = self.encoder(enc)

        if target_ids is not None:
            tgt_in = target_ids[:, :-1]
            tgt_embed = self.embed(tgt_in)
            output = self.decoder(tgt_embed, memory)
            logits = self.classifier(output)
            return logits
        else:
            raise NotImplementedError("Inference decoding not implemented.")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = os.path.expanduser("~/scratch/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv")
    pt_dir = os.path.expanduser("~/data/thesis_storage/tokens/train")

    dataset = VideoToken2GlossDataset(csv_path, pt_dir,
                                      target_channels=60,
                                      target_H=8, target_W=8)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
                            collate_fn=collate_fn, num_workers=2)

    sample = next(iter(dataloader))
    B, T, C, H, W = sample["video_tokens"].shape
    token_dim = C * H * W

    vocab_size = len(dataset.vocab)
    model = Seq2SeqVideoToGloss(token_dim, vocab_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.vocab["<pad>"])

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            vt = batch["video_tokens"].to(device)
            vt = vt.view(vt.size(0), vt.size(1), -1)
            vt = vt / 127.5 - 1  # Normalize

            tgt = batch["target_ids"].to(device)
            logits = model(vt, tgt)
            logits = logits.view(-1, vocab_size)
            labels = tgt[:, 1:].reshape(-1)
            loss = loss_fn(logits, labels)

            if torch.isnan(loss):
                print("❌ NaN loss encountered. Skipping batch.")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/10 — loss: {total_loss/len(dataloader):.6f}")

    torch.save({"model_state_dict": model.state_dict(),
                "vocab": dataset.vocab},
               "video2gloss_seq2seq.pt")
    print("✅ Model saved to video2gloss_seq2seq.pt")

if __name__ == "__main__":
    train()
