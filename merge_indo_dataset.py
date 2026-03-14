import pandas as pd

hoax = pd.read_csv("hf_fake_news_indo.csv")

hoax = hoax[["title", "text", "label"]]
hoax["label"] = 1

factual = pd.read_csv("final_merge_dataset.csv")

factual = factual.rename(columns={
    "Judul": "title",
    "Content": "text"
})

factual = factual[["title", "text"]]
factual["label"] = 0

merged = pd.concat([hoax, factual], ignore_index=True)

merged = merged.sample(frac=1).reset_index(drop=True)

merged.to_csv("hf_fake_news_indo_final.csv", index=False)

print("✅ Dataset merged saved as hf_fake_news_indo_final.csv")
print("Total data:", len(merged))