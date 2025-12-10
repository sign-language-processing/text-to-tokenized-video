import csv
import json
from pathlib import Path


def convert_phoenix_csv_to_json(input_file, output_file):
    items = []

    with open(input_file, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter='|')

        for row in reader:
            en = row["orth"].strip()
            ro = row["translation"].strip()

            # HuggingFace expected format
            items.append({
                "translation": {
                    "en": en,
                    "ro": ro
                }
            })

    with open(output_file, "w", encoding="utf-8") as outfile:
        for entry in items:
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✔ Wrote {len(items)} lines to {output_file}")


# Path to PHOENIX annotation folder
base_path = Path("/scratch/mpanag/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual")

convert_phoenix_csv_to_json(base_path / "PHOENIX-2014-T.train.corpus.csv",
                            "phoenix_translation_train.json")

convert_phoenix_csv_to_json(base_path / "PHOENIX-2014-T.dev.corpus.csv",
                            "phoenix_translation_dev.json")


