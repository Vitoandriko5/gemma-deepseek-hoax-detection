import csv
import time
import random
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel


# =====================================================
# CONFIG
# =====================================================

BASE_MODEL = "google/gemma-2b"
LORA_PATH = "../gemma_fine_tuning/gemma_vito/checkpoint-750"

MODEL_NAME = "Gemma_Vito (Fine-tuned)"

MAX_DATA = 300
SLEEP_TIME = 0.2


# =====================================================
# METRICS
# =====================================================

TP = TN = FP = FN = 0
UNKNOWN = 0


# =====================================================
# DEVICE
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# =====================================================
# LOAD MODEL
# =====================================================

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

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

model.eval()


# =====================================================
# NORMALIZE OUTPUT (ANTI UNKNOWN)
# =====================================================

def normalize(text):

    t = text.lower().strip()

    if "hoax" in t:
        return "Hoax"

    if "factual" in t or "fact" in t or "true" in t:
        return "Factual"

    return "UNKNOWN"


# =====================================================
# ASK MODEL
# =====================================================

def ask_llm(news):

    prompt = f"""
You are a professional fake news classifier.

Classify the news as:

Hoax = false / misleading / fake
Factual = true / reliable

Rules:
- Answer ONLY one word
- No explanation
- No extra text

News:
{news}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(
        out[0],
        skip_special_tokens=True
    )

    result = decoded.split("Answer:")[-1].strip()

    return normalize(result)


# =====================================================
# START TIMER
# =====================================================

start_time = time.time()


# =====================================================
# LOAD DATA
# =====================================================

print("Loading dataset...")

with open("hf_fake_news.csv", newline="", encoding="utf-8") as f:

    reader = list(csv.DictReader(f))
    random.shuffle(reader)


# =====================================================
# MAIN LOOP
# =====================================================

print("\nSTART EVALUATION 🚀\n")

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


    # ================= ETA =================

    elapsed = time.time() - start_time
    avg_time = elapsed / (i + 1)
    remaining = avg_time * (MAX_DATA - (i + 1))

    eta_m = int(remaining // 60)
    eta_s = int(remaining % 60)


    # ================= PRINT =================

    print(
        f"[{i+1:>4}/{MAX_DATA:<4}] | "
        f"True: {true_label:<8} | "
        f"Pred: {prediction:<8} | "
        f"ETA: {eta_m:02d}m {eta_s:02d}s"
    )


    # ================= CONFUSION =================

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


# =====================================================
# FINAL METRICS
# =====================================================

end_time = time.time()

total = TP + TN + FP + FN

accuracy = (TP + TN) / total if total else 0
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0


# =====================================================
# FINAL REPORT
# =====================================================

print("\n" + "=" * 60)
print("===== FINAL EVALUATION (GEMMA_VITO) =====")
print("=" * 60)

print(f"{'Model':<20}: {MODEL_NAME}")
print(f"{'Total Data':<20}: {total}")
print(f"{'Unknown Output':<20}: {UNKNOWN}\n")


print(f"{'Accuracy':<20}: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'Precision':<20}: {precision:.4f} ({precision*100:.2f}%)")
print(f"{'Recall':<20}: {recall:.4f} ({recall*100:.2f}%)")
print(f"{'F1-score':<20}: {f1:.4f} ({f1*100:.2f}%)")


print("\nConfusion Matrix")
print("-" * 60)

print(f"{'TP (Hoax → Hoax)':<40}: {TP}")
print(f"{'TN (Factual → Factual)':<40}: {TN}")
print(f"{'FP (Factual → Hoax)':<40}: {FP}")
print(f"{'FN (Hoax → Factual)':<40}: {FN}")

print("-" * 60)


print(f"\n{'Total Time':<20}: {round(end_time - start_time,2)} seconds\n")