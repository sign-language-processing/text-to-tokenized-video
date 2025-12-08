import pandas as pd
import json

class GlossTokenizer:
    def __init__(self, glosses, min_freq=1):
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.word_freq = {}

        # Count word frequencies
        for gloss in glosses:
            for word in gloss.split():
                self.word_freq[word] = self.word_freq.get(word, 0) + 1

        # Build vocabulary
        self.itos = self.special_tokens + sorted(
            [w for w, f in self.word_freq.items() if f >= min_freq]
        )
        self.stoi = {w: i for i, w in enumerate(self.itos)}

        # Token IDs
        self.pad_id = self.stoi["<PAD>"]
        self.sos_id = self.stoi["<SOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.unk_id = self.stoi["<UNK>"]

    def encode(self, gloss, add_special_tokens=True):
        ids = [self.stoi.get(w, self.unk_id) for w in gloss.split()]
        if add_special_tokens:
            return [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids):
        words = [
            self.itos[i]
            for i in ids
            if i not in (self.pad_id, self.sos_id, self.eos_id)
        ]
        return " ".join(words)

    def __len__(self):
        return len(self.itos)

    def save(self, path="gloss_tokenizer.json"):
        tokenizer_data = {
            "itos": self.itos,
            "stoi": self.stoi
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

# ================================
#            MAIN SCRIPT
# ================================
if __name__ == "__main__":
    csv_path = "video2gloss_overfit.csv"
    df = pd.read_csv(csv_path)

    glosses = df["gloss"].tolist()

    tokenizer = GlossTokenizer(glosses)
    print(f"✅ Loaded {len(glosses)} glosses")
    print(f"🔠 Vocabulary size: {len(tokenizer)}\n")

    print("📋 All glosses:\n")
    for gloss in glosses:
        encoded = tokenizer.encode(gloss)
        decoded = tokenizer.decode(encoded)
        print(f"- Gloss:    {gloss}")
        print(f"  Encoded:  {encoded}")
        print(f"  Decoded:  {decoded}\n")

    tokenizer.save()
    print("💾 Saved tokenizer to gloss_tokenizer.json")
