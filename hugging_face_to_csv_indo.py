from datasets import load_dataset
import pandas as pd

dataset = load_dataset("SEACrowd/id_hoax_news")

print(dataset)

print("Available splits:", dataset.keys())

df = dataset["train"].to_pandas()

print("Columns:", df.columns)

df.to_csv("hf_fake_news_indo.csv", index=False, encoding="utf-8")

print("✅ Saved as hf_fake_news_indo.csv")