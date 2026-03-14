import pandas as pd
import torch
import time
import random

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =====================
# CONFIG
# =====================

MODEL_NAME = "indobenchmark/indobert-base-p1"
DATASET = "hf_fake_news_indo.csv"

MAX_DATA = 10      # ganti: 300 / 1000 / 2000 / 6930
SLEEP_TIME = 0.5   # kecil aja biar ga berat

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


TP = TN = FP = FN = 0
UNKNOWN = 0

start_time = time.time()


# =====================
# LOAD DATA
# =====================

print("Loading dataset...")

df = pd.read_csv(DATASET)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df = df.head(MAX_DATA)

print("Total data:", len(df))


# =====================
# LOAD MODEL
# =====================

print("Loading IndoBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(DEVICE)
model.eval()


# =====================
# EVALUATION LOOP
# =====================

for i, row in df.iterrows():

    title = str(row["title"])
    text = str(row["text"])

    full_text = f"{title}. {text}"

    label = int(row["label"])

    if label == 1:
        true_label = "Hoax"
    else:
        true_label = "Factual"


    # "Prompt" versi Indo (buat dokumentasi, bukan buat model)
    prompt = f"""
Klasifikasikan berita berikut sebagai:

Hoax = berita bohong / menyesatkan
Factual = berita benar / netral

Berita:
{full_text}

Jawaban:
"""

    inputs = tokenizer(
        full_text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(outputs.logits, dim=1).item()


    if pred_id == 1:
        prediction = "Hoax"
    else:
        prediction = "Factual"


    # =====================
    # ETA
    # =====================

    elapsed = time.time() - start_time
    avg_time = elapsed / (i + 1)
    remaining = avg_time * (MAX_DATA - (i + 1))

    eta_min = int(remaining // 60)
    eta_sec = int(remaining % 60)


    print(
        f"[{i+1:>4}/{MAX_DATA:<4}] | "
        f"True: {true_label:<8} | "
        f"Predicted: {prediction:<8} | "
        f"ETA: {eta_min:02d}m {eta_sec:02d}s"
    )


    # =====================
    # CONFUSION MATRIX
    # =====================

    if true_label == "Hoax" and prediction == "Hoax":
        TP += 1

    elif true_label == "Factual" and prediction == "Factual":
        TN += 1

    elif true_label == "Factual" and prediction == "Hoax":
        FP += 1

    elif true_label == "Hoax" and prediction == "Factual":
        FN += 1


    time.sleep(SLEEP_TIME)



# =====================
# FINAL RESULT
# =====================

end_time = time.time()

total = TP + TN + FP + FN

accuracy = (TP + TN) / total if total else 0
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0


print("\n===== FINAL EVALUATION (INDOBERT) =====\n")

print(f"{'Model':<20}: {MODEL_NAME}")
print(f"{'Total Data':<20}: {total}")
print(f"{'Unknown Output':<20}: {UNKNOWN}\n")

print(f"{'Accuracy':<20}: {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"{'Precision':<20}: {precision:.4f}  ({precision*100:.2f}%)")
print(f"{'Recall':<20}: {recall:.4f}  ({recall*100:.2f}%)")
print(f"{'F1-score':<20}: {f1:.4f}  ({f1*100:.2f}%)")

print("\nConfusion Matrix:")
print("-" * 70)

print(f"{'TP (Hoax → Hoax)':<45} | {TP:>5}")
print(f"{'TN (Factual → Factual)':<45} | {TN:>5}")
print(f"{'FP (Factual → Hoax)':<45} | {FP:>5}")
print(f"{'FN (Hoax → Factual)':<45} | {FN:>5}")

print("-" * 70)

print(f"\n{'Total Time':<20}: {round(end_time - start_time, 2)} seconds\n")