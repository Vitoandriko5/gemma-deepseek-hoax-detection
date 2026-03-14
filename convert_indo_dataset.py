import pandas as pd
df = pd.read_csv("dataset_indobert.csv")

print("Columns:", df.columns)

df_new = pd.DataFrame()

df_new["title"] = df["judul"]
df_new["text"] = df["narasi"]
df_new["label"] = df["label"]

df_new.to_csv("hf_fake_news_indo.csv", index=False)

print("✅ Saved as hf_fake_news_indo.csv")
print(df_new.head())