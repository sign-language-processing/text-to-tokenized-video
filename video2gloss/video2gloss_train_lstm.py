import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import wandb

# === CONFIG ===
DATA_PATH = "video_to_gloss_train.jsonl"
VAL_PATH = "video_to_gloss_val.jsonl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
PATIENCE = 30
HIDDEN_SIZE = 256
EMBED_SIZE = 128
PROJECT_NAME = "video2gloss"

# === INIT WANDB ===
wandb.init(project=PROJECT_NAME, config={
    "batch_size": BATCH_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "embed_size": EMBED_SIZE,
    "patience": PATIENCE,
    "device": str(DEVICE)
})

# === DATASET ===
class GlossDataset(Dataset):
    def __init__(self, path, vocab=None):
        with open(path, 'r') as f:
            self.data = [json.loads(line) for line in f]

        glosses = [g for d in self.data for g in d["target_glosses"]]

        if vocab is None:
            self.vocab = {
                "<PAD>": 0,
                "<SOS>": 1,
                "<EOS>": 2,
                "<UNK>": 3
            }
            for g in sorted(set(glosses)):
                self.vocab[g] = len(self.vocab)
        else:
            self.vocab = vocab

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = torch.tensor(item["input_codes"], dtype=torch.float)
        gloss = ["<SOS>"] + item["target_glosses"] + ["<EOS>"]
        y = torch.tensor([self.vocab.get(g, self.vocab["<UNK>"]) for g in gloss], dtype=torch.long)
        return x, y

def collate_fn(batch):
    x_batch, y_batch = zip(*batch)
    x_padded = nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
    y_padded = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=0)
    return x_padded, y_padded

# === MODEL ===
class VideoToGloss(nn.Module):
    def __init__(self, input_dim, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_size, batch_first=True, bidirectional=True)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.decoder = nn.LSTM(embed_size, hidden_size * 2, batch_first=True)
        self.output = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, y):
        _, (h, c) = self.encoder(x)
        h = torch.cat((h[0], h[1]), dim=-1).unsqueeze(0)
        c = torch.cat((c[0], c[1]), dim=-1).unsqueeze(0)
        y_embed = self.embedding(y[:, :-1])
        out, _ = self.decoder(y_embed, (h, c))
        logits = self.output(out)
        return logits

# === BLEU EVALUATION ===
def compute_bleu(model, dataset, num_samples=20):
    model.eval()
    smooth = SmoothingFunction().method4
    scores = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(DEVICE)

            _, (h, c) = model.encoder(x)
            h = torch.cat((h[0], h[1]), dim=-1).unsqueeze(0)
            c = torch.cat((c[0], c[1]), dim=-1).unsqueeze(0)

            token = torch.tensor([[dataset.vocab["<SOS>"]]], device=DEVICE)
            decoded = []

            for _ in range(30):
                emb = model.embedding(token)
                out, (h, c) = model.decoder(emb, (h, c))
                pred = model.output(out[:, -1])
                token = pred.argmax(dim=-1, keepdim=True)
                word = dataset.inv_vocab[token.item()]
                if word == "<EOS>":
                    break
                decoded.append(word)

            target = [dataset.inv_vocab[t.item()] for t in y[1:-1]]
            bleu = sentence_bleu([target], decoded, smoothing_function=smooth)
            scores.append(bleu)

    return sum(scores) / len(scores)

# === TRAINING ===
def train():
    train_data = GlossDataset(DATA_PATH)
    val_data = GlossDataset(VAL_PATH, vocab=train_data.vocab)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = VideoToGloss(
        input_dim=384,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=len(train_data.vocab)
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_bleu = 0
    patience_counter = 0
    epoch = 0

    while True:
        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x, y)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y[:, 1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        bleu = compute_bleu(model, val_data)

        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "bleu": bleu
        })

        print(f"\nEpoch {epoch+1} Loss: {avg_loss:.4f}")
        print(f"BLEU after epoch {epoch+1}: {bleu:.4f}")

        if bleu > best_bleu:
            best_bleu = bleu
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("Model improved and saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

        epoch += 1

    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(train_data.vocab, f, indent=2)

    print("Training completed.")
    wandb.finish()

if __name__ == "__main__":
    train()