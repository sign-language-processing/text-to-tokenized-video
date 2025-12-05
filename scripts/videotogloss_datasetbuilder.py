import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class PhoenixVideoTokenGlossDataset(Dataset):
    def __init__(self, csv_path, pt_dir, max_len=64, vocab=None):
        self.df = pd.read_csv(csv_path, sep="|")
        self.pt_dir = pt_dir
        self.max_len = max_len

        # Build vocabulary
        if vocab is None:
            all_glosses = set()
            for gloss_seq in self.df["orth"]:
                all_glosses.update(gloss_seq.strip().split())
            self.vocab = {g: i + 1 for i, g in enumerate(sorted(all_glosses))}
            self.vocab["<pad>"] = 0
        else:
            self.vocab = vocab

        self.id2gloss = {i: g for g, i in self.vocab.items()}

        # Keep only valid samples
        self.df = self.df[self.df["name"].apply(lambda n: os.path.isfile(os.path.join(pt_dir, f"{n}.pt")))].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["name"]
        token_path = os.path.join(self.pt_dir, f"{name}.pt")
        video_tokens = torch.load(token_path, map_location="cpu")

        # unpack tuple if needed
        if isinstance(video_tokens, tuple):
            video_tokens = video_tokens[0]

        print("DEBUG shape:", video_tokens.shape)

        # slice along time dimension
        video_tokens = video_tokens[:self.max_len]

        # correct padding for ANY shape
        pad_len = self.max_len - video_tokens.shape[0]
        if pad_len > 0:
            pad_shape = (pad_len,) + video_tokens.shape[1:]
            pad = torch.zeros(pad_shape, dtype=video_tokens.dtype)
            video_tokens = torch.cat([video_tokens, pad], dim=0)

        # gloss → ID mapping
        glosses = row["orth"].split()
        gloss_ids = [self.vocab.get(g, 0) for g in glosses[:self.max_len]]
        gloss_ids += [0] * (self.max_len - len(gloss_ids))

        return {
            "input_ids": video_tokens.long(),
            "labels": torch.tensor(gloss_ids, dtype=torch.long),
            "attention_mask": (video_tokens != 0).long(),
        }


# Use ~ safely and ensure the paths are valid
csv_path = os.path.expanduser("~/scratch/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv")
pt_dir = os.path.expanduser("~/data/thesis_storage/tokens/train")

# Create dataset
dataset = PhoenixVideoTokenGlossDataset(csv_path=csv_path, pt_dir=pt_dir, max_len=64)
print(f"✅ Loaded {len(dataset)} samples.")

# Inspect first item
item = dataset[0]
print("Input shape:", item["input_ids"].shape)
print("First 10 label IDs:", item["labels"][:10])
print("First 10 attention values:", item["attention_mask"][:10])
