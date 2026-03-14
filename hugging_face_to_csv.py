from datasets import load_dataset
import pandas as pd

dataset = load_dataset("mohammadjavadpirhadi/fake-news-detection-dataset-english")
df_train = dataset["train"].to_pandas()
print("Columns:", df_train.columns)
df_train.to_csv("hf_fake_news.csv", index=False)
print("✅ CSV saved as hf_fake_news.csv")