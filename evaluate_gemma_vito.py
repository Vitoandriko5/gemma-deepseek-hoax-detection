import csv
import time
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# =========================
# CONFIG
# =========================

BASE_MODEL = "google/gemma-2b"
LORA_PATH = "../gemma_fine_tuning/gemma_vito/checkpoint-750"

MODEL_NAME = "Gemma_Vito (Fine-tuned)"

MAX_DATA = 300
SLEEP_TIME = 0.2


# =========================
# METRICS
# =========================

TP = TN = FP = FN = 0
UNKNOWN = 0


# =========================
# DEVICE
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# =========================
# LOAD MODEL
# =========================

print("Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()


# =========================
# NORMALIZE OUTPUT
# =========================

def normalize(text):

    t = text.lower()

    if "hoax" in t:
        return "Hoax"

    if "fakta" in t or "factual" in t:
        return "Factual"

    return "UNKNOWN"


# =========================
# ASK MODEL
# =========================

def ask_llm(news):

    prompt = f"""
You are a professional fake news classification system.

Rules:
- If the news is false, misleading, or unverified → answer: Hoax
- If the news is accurate and reliable → answer: Factual
- Do NOT explain
- Do NOT add extra words

Answer ONLY:
Hoax
or
Factual

News:
{news}

Answer:
"""


#     prompt = f"""
# Anda adalah sistem klasifikasi berita.

# Tugas Anda:
# Jawab hanya dengan satu kata: Hoax atau Fakta.

# Berita:
# {news}

# Jawaban:
# """

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    result = decoded.split("Jawaban:")[-1].strip()

    return normalize(result)


# =========================
# START
# =========================

start_time = time.time()


# =========================
# LOAD DATA
# =========================

with open("hf_fake_news.csv", newline="", encoding="utf-8") as f:

    reader = list(csv.DictReader(f))
    random.shuffle(reader)


# =========================
# MAIN LOOP
# =========================

for i, row in enumerate(reader):

    if i >= MAX_DATA:
        break

    title = row["title"]
    text = row["text"]

    news = f"{title}. {text}"

    label = row["label"].strip()

    if label == "1":
        true_label = "Hoax"
    else:
        true_label = "Factual"


    prediction = ask_llm(news)

    time.sleep(SLEEP_TIME)


    # =========================
    # ETA
    # =========================

    elapsed = time.time() - start_time
    avg = elapsed / (i + 1)
    remain = avg * (MAX_DATA - (i + 1))

    eta_m = int(remain // 60)
    eta_s = int(remain % 60)


    # =========================
    # PROGRESS
    # =========================

    print(
        f"[{i+1:>4}/{MAX_DATA:<4}] | "
        f"True: {true_label:<8} | "
        f"Pred: {prediction:<8} | "
        f"ETA: {eta_m:02d}m {eta_s:02d}s"
    )


    # =========================
    # CONFUSION MATRIX
    # =========================

    if prediction == "UNKNOWN":
        UNKNOWN += 1
        continue


    if true_label == "Hoax" and prediction == "Hoax":
        TP += 1

    elif true_label == "Factual" and prediction == "Factual":
        TN += 1

    elif true_label == "Factual" and prediction == "Hoax":
        FP += 1

    elif true_label == "Hoax" and prediction == "Factual":
        FN += 1


# =========================
# FINAL METRICS
# =========================

end_time = time.time()

total = TP + TN + FP + FN

accuracy = (TP + TN) / total if total else 0
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0


# =========================
# FINAL OUTPUT
# =========================

print("\n===== FINAL EVALUATION (GEMMA_VITO) =====\n")

print(f"{'Model':<20}: {MODEL_NAME}")
print(f"{'Total Data':<20}: {total}")
print(f"{'Unknown Output':<20}: {UNKNOWN}\n")


print(f"{'Accuracy':<20}: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'Precision':<20}: {precision:.4f} ({precision*100:.2f}%)")
print(f"{'Recall':<20}: {recall:.4f} ({recall*100:.2f}%)")
print(f"{'F1-score':<20}: {f1:.4f} ({f1*100:.2f}%)")


print("\nConfusion Matrix:")
print("-" * 70)

print(f"{'TP (Hoax → Hoax)':<45} | {TP:>5}")
print(f"{'TN (Fakta → Fakta)':<45} | {TN:>5}")
print(f"{'FP (Fakta → Hoax)':<45} | {FP:>5}")
print(f"{'FN (Hoax → Fakta)':<45} | {FN:>5}")

print("-" * 70)


print(f"\n{'Total Time':<20}: {round(end_time - start_time,2)} seconds\n")