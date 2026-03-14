import pandas as pd

# Load hoax dataset
data = pd.read_csv("hf_fake_news_indo.csv")

# Load factual dataset (media)
fact_media = pd.read_csv("final_merge_2699.csv")

# Rename kolom media
fact_media = fact_media.rename(columns={
    "Judul": "title",
    "Content": "text"
})

# Tambah label factual = 0
fact_media["label"] = 0

fact_media = fact_media[["title", "text", "label"]]
data = data[["title", "text", "label"]]

# Pisahkan hoax & factual dari hf
hoax = data[data["label"] == 1]
fact_hf = data[data["label"] == 0]

print("Hoax:", len(hoax))
print("Factual HF:", len(fact_hf))
print("Factual Media:", len(fact_media))

# Gabung semua factual
fact_all = pd.concat([fact_hf, fact_media])

# Cari jumlah minimum
min_size = min(len(hoax), len(fact_all))

# Sampling seimbang
hoax_sample = hoax.sample(n=min_size, random_state=42)
fact_sample = fact_all.sample(n=min_size, random_state=42)

# Merge final
final = pd.concat([hoax_sample, fact_sample])

# Shuffle
final = final.sample(frac=1, random_state=42)

# Save
final.to_csv("dataset_final_indo_balanced.csv", index=False)

print("\nSaved: dataset_final_indo_balanced.csv")
print("Total:", len(final))
print("Label counts:")
print(final["label"].value_counts())