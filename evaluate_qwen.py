import csv
import ollama
import time
import random

model_name = "qwen2.5:3b"

MAX_DATA = 2000
SLEEP_TIME = 0.5

TP = TN = FP = FN = 0
UNKNOWN = 0   

start_time = time.time()

def normalize_prediction(text):

    t = text.lower().strip()

    if t == "hoax":
        return "Hoax"

    if t == "factual":
        return "Factual"

    if "hoax" in t or "fake" in t or "false" in t:
        return "Hoax"

    if "factual" in t or "fact" in t or "true" in t or "real" in t:
        return "Factual"

    first = t.split()[0] if t else ""

    if first.startswith("h"):
        return "Hoax"

    if first.startswith("f"):
        return "Factual"

    return "Hoax"

with open("hf_fake_news.csv", newline="", encoding="utf-8") as csvfile:

    reader = list(csv.DictReader(csvfile))
    # random.shuffle(reader)

    for i, row in enumerate(reader):

        if i >= MAX_DATA:
            break

        title = row["title"]
        text = row["text"]

        full_text = f"{title}. {text}"

        label = row["label"].strip()

        if label == "1":
            true_label = "Hoax"
        else:
            true_label = "Factual"

        prompt = f"""
You are a strict fake news classifier.

Classify the news into ONE of two categories:

Hoax = contains false claims, misinformation, or unverifiable statements.
Factual = contains verifiable, credible, or neutral information.

You MUST choose exactly one label.

Do not avoid classification.

News:
{full_text}

Answer:
"""

        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0,
                "num_predict": 2,
                "stop": ["\n", ".", ",", " "]
            }
        )

        time.sleep(SLEEP_TIME)

        raw_output = response["message"]["content"].strip()

        prediction = normalize_prediction(raw_output)

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

        if true_label == "Hoax" and prediction == "Hoax":
            TP += 1

        elif true_label == "Factual" and prediction == "Factual":
            TN += 1

        elif true_label == "Factual" and prediction == "Hoax":
            FP += 1

        elif true_label == "Hoax" and prediction == "Factual":
            FN += 1


end_time = time.time()

total = TP + TN + FP + FN

accuracy = (TP + TN) / total if total else 0
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

print("\n===== FINAL EVALUATION (QWEN) =====\n")

print(f"{'Model':<20}: {model_name}")
print(f"{'Total Data':<20}: {total}")
print(f"{'Unknown Output':<20}: {UNKNOWN}\n")

print(f"{'Accuracy':<20}: {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"{'Precision':<20}: {precision:.4f}  ({precision*100:.2f}%)")
print(f"{'Recall':<20}: {recall:.4f}  ({recall*100:.2f}%)")
print(f"{'F1-score':<20}: {f1:.4f}  ({f1*100:.2f}%)")

print("\nConfusion Matrix:")
print("-" * 70)

print(f"{'TP (Hoax news → Hoax predicted)':<45} | {TP:>5}")
print(f"{'TN (Factual news → Factual predicted)':<45} | {TN:>5}")
print(f"{'FP (Factual news → Hoax predicted)':<45} | {FP:>5}")
print(f"{'FN (Hoax news → Factual predicted)':<45} | {FN:>5}")

print("-" * 70)

print(f"\n{'Total Time':<20}: {round(end_time - start_time, 2)} seconds\n")