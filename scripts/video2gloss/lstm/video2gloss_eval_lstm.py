#!/usr/bin/env python3
import json

import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.utils.data import DataLoader, Dataset
from video2gloss_train_lstm import VideoToGloss, collate_fn  # reuse model + collate

# === CONFIG ===
MODEL_PATH = "best_model.pt"
VOCAB_PATH = "vocab.json"
TEST_PATH = "video_to_gloss_test.jsonl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 634  # how many samples to evaluate

# === LOAD VOCAB ===
with open(VOCAB_PATH, encoding="utf-8") as f:
    vocab = json.load(f)

vocab = {w: int(i) for w, i in vocab.items()}
inv_vocab = {i: w for w, i in vocab.items()}


# === LOAD DATASET ===
class TestGlossDataset(Dataset):
    def __init__(self, path, vocab):
        with open(path, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]
        self.vocab = vocab
        self.inv_vocab = {i: w for w, i in vocab.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = torch.tensor(item["input_codes"], dtype=torch.float)
        gloss = ["<SOS>"] + item["target_glosses"] + ["<EOS>"]
        y = torch.tensor([self.vocab.get(g, self.vocab["<UNK>"]) for g in gloss], dtype=torch.long)
        return x, y


# === LOAD TEST SET ===
test_data = TestGlossDataset(TEST_PATH, vocab)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

# === LOAD MODEL ===
model = VideoToGloss(input_dim=384, embed_size=128, hidden_size=256, vocab_size=len(vocab))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === EVALUATE ===
smooth = SmoothingFunction().method4
bleu_scores = []

print("\nRunning Evaluation on Best Model (BLEU-based Early Stopping)\n")

for i, (x, y) in enumerate(test_loader):
    if i >= NUM_SAMPLES:
        break

    x = x.to(DEVICE)

    with torch.no_grad():
        # Encode
        _, (h, c) = model.encoder(x)
        h = torch.cat((h[0], h[1]), dim=-1).unsqueeze(0)
        c = torch.cat((c[0], c[1]), dim=-1).unsqueeze(0)

        # Decode
        token = torch.tensor([[vocab["<SOS>"]]], device=DEVICE)
        decoded = []

        for _ in range(30):
            emb = model.embedding(token)
            out, (h, c) = model.decoder(emb, (h, c))
            pred = model.output(out[:, -1])
            token = pred.argmax(dim=-1, keepdim=True)
            word = inv_vocab[token.item()]
            if word == "<EOS>":
                break
            decoded.append(word)

        target = [inv_vocab[t.item()] for t in y[0][1:-1]]
        bleu = sentence_bleu([target], decoded, smoothing_function=smooth)
        bleu_scores.append(bleu)

        print(f"Sample {i + 1}")
        print(f"Target   : {' '.join(target)}")
        print(f"Predicted: {' '.join(decoded)}")
        print(f"BLEU     : {bleu:.4f}\n")

print(f"\nAverage BLEU on Test Set (n={NUM_SAMPLES}): {sum(bleu_scores) / len(bleu_scores):.4f}")
