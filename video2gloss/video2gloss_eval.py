import json
import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.data import Dataset, DataLoader
from video2gloss_train_lstm import VideoToGloss, collate_fn  # reuse model + collate

# === CONFIG ===
MODEL_PATH = "best_model.pt"
VOCAB_PATH = "vocab.json"
DEV_PATH = "video_to_gloss_dev.jsonl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 516  # how many samples to evaluate

# === LOAD VOCAB ===
with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

inv_vocab = {int(i): w for w, i in vocab.items()}
vocab = {w: int(i) for w, i in vocab.items()}


# === LOAD DATASET ===
class DevGlossDataset(Dataset):
    def __init__(self, path, vocab):
        with open(path, 'r', encoding='utf-8') as f:
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


# === LOAD DEV SET ===
dev_data = DevGlossDataset(DEV_PATH, vocab)
dev_loader = DataLoader(dev_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

# === LOAD MODEL ===
model = VideoToGloss(input_dim=384, embed_size=128, hidden_size=256, vocab_size=len(vocab))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === EVALUATE ===
smooth = SmoothingFunction().method4
bleu_scores = []

print("\nRunning Evaluation on Best Model (BLEU-based Early Stopping)\n")

for i, (x, y) in enumerate(dev_loader):
    if i >= NUM_SAMPLES:
        break

    x = x.to(DEVICE)
    # After encoder
    _, (h, c) = model.encoder(x)  # h and c are [2, batch, hidden]
    h = torch.cat((h[0], h[1]), dim=-1).unsqueeze(0)  # → [1, batch, hidden*2]
    c = torch.cat((c[0], c[1]), dim=-1).unsqueeze(0)  # same

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

    print(f"Sample {i+1}")
    print(f"Target   : {' '.join(target)}")
    print(f"Predicted: {' '.join(decoded)}")
    print(f"BLEU     : {bleu:.4f}\n")

print(f"\nAverage BLEU on Dev Set (n={NUM_SAMPLES}): {sum(bleu_scores)/len(bleu_scores):.4f}")
