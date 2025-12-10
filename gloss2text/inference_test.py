import json
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# Load model + tokenizer (local path)
model_path = "/home/mpanag/thesis/text-to-tokenized-video/scripts/models/phoenix_translation"
tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
model = MarianMTModel.from_pretrained(model_path, local_files_only=True)

dev_file = Path("phoenix_translation_dev.json")
predictions = []
references = []

# Iterate all dev examples
with open(dev_file, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        gloss = obj["translation"]["en"]
        ref = obj["translation"]["ro"]

        # Tokenize & generate
        inputs = tokenizer(gloss, return_tensors="pt", padding=True, truncation=True)
        output_ids = model.generate(**inputs, num_beams=4, max_length=512)[0]
        pred = tokenizer.decode(output_ids, skip_special_tokens=True)

        predictions.append(pred.strip())
        references.append(ref.strip())

# Save to files (optional)
with open("predictions.txt", "w", encoding="utf-8") as f:
    for p in predictions:
        f.write(p + "\n")

with open("references.txt", "w", encoding="utf-8") as f:
    for r in references:
        f.write(r + "\n")

# Compute metrics
import sacrebleu

bleu = sacrebleu.corpus_bleu(predictions, [references])
chrf = sacrebleu.corpus_chrf(predictions, [references])

print("BLEU:", bleu.score)
print("chrF:", chrf.score)

# METEOR (optional, needs nltk)
from nltk.translate.meteor_score import meteor_score

meteor = sum(
    meteor_score([r.split()], p.split()) for p, r in zip(predictions, references)
) / len(references)

print("METEOR (avg):", meteor)
#
# from transformers import MarianMTModel, MarianTokenizer
# from pathlib import Path
# import json
#
# model_path = Path("/tmp/tst-translation").resolve()
# tokenizer = MarianTokenizer.from_pretrained(str(model_path))
# model = MarianMTModel.from_pretrained(str(model_path))
#
# # Load dev samples
# dev_file = Path("phoenix_translation_dev.json")
# examples = []
# with open(dev_file, encoding="utf-8") as f:
#     for line in f:
#         data = json.loads(line)
#         examples.append(data["translation"])
#         if len(examples) >= 10:
#             break
#
# print(" Qualitative Evaluation of Translations:\n")
# for i, example in enumerate(examples, 1):
#     gloss = example["en"]
#     reference = example["ro"]
#
#     inputs = tokenizer(gloss, return_tensors="pt", padding=True, truncation=True)
#     translated = model.generate(**inputs, num_beams=4, max_length=512)
#     prediction = tokenizer.decode(translated[0], skip_special_tokens=True)
#
#     print(f"--- Example {i} ---")
#     print(f" Gloss:      {gloss}")
#     print(f" Reference:  {reference}")
#     print(f" Prediction: {prediction}")
#     print()
