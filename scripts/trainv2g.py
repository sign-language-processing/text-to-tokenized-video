import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
import wandb

# ================================
# 1) Load Gloss Tokenizer
# ================================
with open("gloss_tokenizer.json", "r", encoding="utf-8") as f:
    tok = json.load(f)

itos = tok["itos"]
stoi = tok["stoi"]

PAD_ID = stoi["<PAD>"]
SOS_ID = stoi["<SOS>"]
EOS_ID = stoi["<EOS>"]
UNK_ID = stoi["<UNK>"]

def encode_gloss(gloss):
    ids = [stoi.get(w, UNK_ID) for w in gloss.split()]
    return [SOS_ID] + ids + [EOS_ID]


# ================================
# 2) Dataset
# ================================
class Video2GlossDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.samples = []

        for _, row in df.iterrows():
            token_path = row["token_path"]
            gloss = row["gloss"]

            try:
                data = torch.load(token_path, map_location="cpu")
            except Exception:
                continue

            if isinstance(data, tuple):
                _, codes = data
            else:
                codes = data

            self.samples.append((codes.to(torch.float32), encode_gloss(gloss)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        codes, gloss_ids = self.samples[idx]
        return codes, torch.tensor(gloss_ids, dtype=torch.long)


# ================================
# 3) Collate function
# ================================
def collate_fn(batch):
    videos, glosses = zip(*batch)

    max_T = max(v.shape[2] for v in videos)
    padded_videos = []
    for v in videos:
        pad_len = max_T - v.shape[2]
        if pad_len > 0:
            pad = torch.zeros((1, v.shape[1], pad_len, v.shape[3], v.shape[4]))
            v = torch.cat([v, pad], dim=2)
        padded_videos.append(v)

    video_batch = torch.cat(padded_videos, dim=0)  # [B, 6, T, 8, 8]

    max_L = max(g.size(0) for g in glosses)
    gloss_batch = torch.full((len(glosses), max_L), PAD_ID, dtype=torch.long)
    for i, g in enumerate(glosses):
        gloss_batch[i, :g.size(0)] = g

    return video_batch, gloss_batch


# ================================
# 4) Simple Transformer Model
# ================================
class SimpleV2GModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, nhead=4, layers=2):
        super().__init__()

        self.proj = nn.Linear(6 * 8 * 8, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)        # [B, T, 6, 8, 8]
        x = x.reshape(B, T, C * H * W)      # [B, T, 384]
        x = self.proj(x)                   # [B, T, hidden]
        x = self.encoder(x)                # [B, T, hidden]
        logits = self.decoder(x)           # [B, T, vocab]
        return logits


# ================================
# 5) Training utils
# ================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for videos, glosses in loader:
        videos = videos.to(device)
        glosses = glosses.to(device)

        optimizer.zero_grad()
        logits = model(videos)
        B, T_vid, V = logits.shape
        L = glosses.shape[1]

        logits = logits[:, :L, :]

        loss = criterion(logits.reshape(-1, V), glosses.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=-1)
        mask = glosses != PAD_ID
        total_correct += (preds == glosses).masked_select(mask).sum().item()
        total_tokens += mask.sum().item()

        total_loss += loss.item()

    acc = total_correct / total_tokens if total_tokens > 0 else 0
    return total_loss / len(loader), acc


def greedy_decode(model, videos, device):
    model.eval()
    videos = videos.to(device)
    with torch.no_grad():
        logits = model(videos)
        pred_ids = logits.argmax(-1).cpu().tolist()
    return pred_ids


# ================================
# 6) MAIN TRAINING LOOP
# ================================
def main():
    wandb.init(project="v2g-overfit", name="small-debug-run", mode="online")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Video2GlossDataset("video2gloss_overfit.csv")
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

    model = SimpleV2GModel(vocab_size=len(itos)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for ep in range(200):
        loss, acc = train_one_epoch(model, loader, optimizer, criterion, device)
        wandb.log({"epoch": ep, "loss": loss, "accuracy": acc})
        if ep % 10 == 0:
            print(f"Epoch {ep}, loss {loss:.4f}, acc {acc:.4f}")

    print("\n=== Inference on train set ===")
    eval_loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)
    videos, glosses = next(iter(eval_loader))
    preds = greedy_decode(model, videos, device)

    for i, pred in enumerate(preds):
        raw_ids = pred
        decoded = [itos[t] for t in raw_ids if t not in (PAD_ID, SOS_ID, EOS_ID)]
        expected = [itos[t.item()] for t in glosses[i] if t.item() not in (PAD_ID, SOS_ID, EOS_ID)]
        print(f"{i} raw IDs → {raw_ids}\n{i} decoded → {' '.join(decoded)}\n    Expected IDs: {expected}")

if __name__ == "__main__":
    main()
