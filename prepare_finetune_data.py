import csv
import json

INPUT_FILE = "hf_fake_news.csv"
OUTPUT_FILE = "gemma_finetune.jsonl"

MAX_DATA = 2000

count = 0

with open(INPUT_FILE, newline="", encoding="utf-8") as csvfile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as jsonlfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

        if count >= MAX_DATA:
            break

        title = row["title"].strip()
        text = row["text"].strip()
        label = row["label"].strip()

        if label == "1":
            answer = "Hoax"
        else:
            answer = "Factual"

        news = f"{title}. {text}"

        data = {
            "prompt": f"""
Classify the following news as Hoax or Factual.

News:
{news}

Answer:
""".strip(),

            "completion": f" {answer}"
        }

        jsonlfile.write(json.dumps(data, ensure_ascii=False) + "\n")

        count += 1


print(f"Done! Generated {count} samples in {OUTPUT_FILE}")