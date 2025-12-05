"""
This script trains a baseline Video-to-Gloss Transformer model using discrete
Cosmos video tokens extracted from the PHOENIX-2014-T sign language dataset.
It loads per-frame token tensors, normalizes and pads them to a fixed size,
pairs them with gloss annotations, and trains a lightweight transformer encoder
to predict a gloss sequence for each video. The final trained model and gloss
vocabulary are saved as `video2gloss_model.pt` for later evaluation or inference.
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class PhoenixVideoTokenGlossDataset(Dataset):
    def __init__(self, csv_path, pt_dir, max_len=64, target_shape=(1, 12, 8), vocab=None):
        self.df = pd.read_csv(csv_path, sep="|")
        self.pt_dir = pt_dir
        self.max_len = max_len
        self.target_shape = target_shape  # (C, H, W)

        if vocab is None:
            all_glosses = set()
            for gloss_seq in self.df["orth"]:
                all_glosses.update(gloss_seq.strip().split())
            self.vocab = {g: i + 1 for i, g in enumerate(sorted(all_glosses))}
            self.vocab["<pad>"] = 0
        else:
            self.vocab = vocab

        self.df = self.df[self.df["name"].apply(
            lambda n: os.path.isfile(os.path.join(pt_dir, f"{n}.pt"))
        )].reset_index(drop=True)

    def fix_spatial(self, x):
        C, H, W = self.target_shape
        xC, xH, xW = x.shape

        if xC < C:
            padC = torch.zeros((C - xC, xH, xW), dtype=x.dtype)
            x = torch.cat([x, padC], dim=0)
        x = x[:C]

        if xH < H:
            padH = torch.zeros((C, H - xH, xW), dtype=x.dtype)
            x = torch.cat([x, padH], dim=1)
        x = x[:, :H]

        if xW < W:
            padW = torch.zeros((C, H, W - xW), dtype=x.dtype)
            x = torch.cat([x, padW], dim=2)
        x = x[:, :, :W]

        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["name"]
        x = torch.load(os.path.join(self.pt_dir, f"{name}.pt"), map_location="cpu")
        if isinstance(x, tuple):
            x = x[0]

        T = min(x.shape[0], self.max_len)
        x = x[:T]

        frames = [self.fix_spatial(x[i]) for i in range(T)]
        x = torch.stack(frames).float() / 255.0  # Normalize if values in 0-255

        if T < self.max_len:
            pad = torch.zeros((self.max_len - T,) + x.shape[1:], dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)

        gloss_ids = [self.vocab.get(g, 0) for g in row["orth"].split()[:self.max_len]]
        gloss_ids += [0] * (self.max_len - len(gloss_ids))

        return {
            "input_tokens": x,
            "labels": torch.tensor(gloss_ids, dtype=torch.long)
        }

    def __len__(self):
        return len(self.df)

def collate_fn(batch):
    tokens = [b["input_tokens"] for b in batch]
    labels = [b["labels"] for b in batch]
    shape0 = tokens[0].shape
    assert all(t.shape == shape0 for t in tokens), "Inconsistent token shapes in batch"

    return {
        "input_tokens": torch.stack(tokens),
        "labels": torch.stack(labels)
    }

class VideoToGlossModel(nn.Module):
    def __init__(self, token_feature_dim, hidden_dim, vocab_size):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.proj = nn.Linear(token_feature_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        x = x.view(B, T, -1)
        x = self.proj(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)
        return self.classifier(x)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = os.path.expanduser("~/scratch/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv")
    pt_dir = os.path.expanduser("~/data/thesis_storage/tokens/train")

    dataset = PhoenixVideoTokenGlossDataset(csv_path=csv_path, pt_dir=pt_dir, max_len=64)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=2)

    example = next(iter(dataloader))
    input_tokens = example["input_tokens"]
    B, T, C, H, W = input_tokens.shape
    token_feature_dim = H * W * C

    vocab_size = len(dataset.vocab)
    model = VideoToGlossModel(token_feature_dim=token_feature_dim, hidden_dim=256, vocab_size=vocab_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(5):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            tokens = batch["input_tokens"].to(device)
            labels = batch["labels"].to(device)

            logits = model(tokens)
            logits = logits.view(-1, vocab_size)
            labels = labels.view(-1)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/5 — loss: {total_loss / len(dataloader):.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": dataset.vocab
    }, "video2gloss_model.pt")
    print("✅ Training complete — model saved to video2gloss_model.pt")

if __name__ == "__main__":
    train()
