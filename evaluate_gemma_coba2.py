import csv
import ollama
import time
import random

model_name = "gemma3:4b"
TP = TN = FP = FN = 0
MAX_DATA = 2000          
SLEEP_TIME = 0.5         


start_time = time.time()

with open("hf_fake_news.csv", newline="", encoding="utf-8") as csvfile:

    # reader = csv.DictReader(csvfile)

    reader = list(csv.DictReader(csvfile))
    random.shuffle(reader)

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
You are an expert in detecting fake news.

Analyze the news carefully.
If the news contains misleading, false, or unverified information, classify it as Hoax.
If it is reliable and factual, classify it as Factual.

Answer ONLY:
Hoax or Factual

News: {full_text}
"""

        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0, "num_predict": 3
            }
        )

        time.sleep(SLEEP_TIME)

        prediction = response["message"]["content"].strip()

        # ETA CALCULATION

        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (MAX_DATA - (i + 1))

        eta_min = int(remaining // 60)
        eta_sec = int(remaining % 60)

        progress = (i + 1) / MAX_DATA * 100

        # ========================
        # PRINT PROGRESS
        # ========================
        # print(f"[{i+1}/{MAX_DATA} | {progress:.2f}%] "
        # print(f"[{i+1}/{MAX_DATA}] "
        #       f"True: {true_label} | Predicted: {prediction} "
        #       f"| ETA: {eta_min}m {eta_sec}s")
        
        print(
    f"[{i+1:>4}/{MAX_DATA:<4}] | "
    f"True: {true_label:<8} | "
    f"Predicted: {prediction:<8} | "
    f"ETA: {eta_min:02d}m {eta_sec:02d}s"
)

        # ========================
        # CONFUSION MATRIX
        # ========================
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

accuracy = (TP + TN) / total
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0


print("\n===== FINAL EVALUATION (GEMMA) =====\n")

# print(f"Model     : {model_name}")
# print(f"Total Data: {total}")

print(f"{'Model':<20}: {model_name}")
print(f"{'Total Data':<20}: {total}\n")

# print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
# print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
# print(f"Recall   : {recall:.4f} ({recall*100:.2f}%)")
# print(f"F1-score : {f1:.4f} ({f1*100:.2f}%)")

# print(f"Accuracy : {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall   : {recall:.4f}")
# print(f"F1-score : {f1:.4f}")

print(f"{'Accuracy':<20}: {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"{'Precision':<20}: {precision:.4f}  ({precision*100:.2f}%)")
print(f"{'Recall':<20}: {recall:.4f}  ({recall*100:.2f}%)")
print(f"{'F1-score':<20}: {f1:.4f}  ({f1*100:.2f}%)")

# print("\nConfusion Matrix:")
# print(f"TP (Hoax news → Hoax predicted): {TP}")
# print(f"TN (Factual news → Factual predicted): {TN}")
# print(f"FP (Factual news → Hoax predicted) predicted: {FP}")
# print(f"FN (Hoax news→ Factual predicted): {FN}")

# print("\nConfusion Matrix:")
# print(f"{'TP (Hoax news → Hoax predicted)':<30}: {TP}")
# print(f"{'TN (Factual news → Factual predicted)':<30}: {TN}")
# print(f"{'FP (Factual news → Hoax predicted)':<30}: {FP}")
# print(f"{'FN (Hoax news → Factual predicted)':<30}: {FN}")

print("\nConfusion Matrix:")
print("-" * 70)

print(f"{'TP (Hoax news → Hoax predicted)':<45} | {TP:>5}")
print(f"{'TN (Factual news → Factual predicted)':<45} | {TN:>5}")
print(f"{'FP (Factual news → Hoax predicted)':<45} | {FP:>5}")
print(f"{'FN (Hoax news → Factual predicted)':<45} | {FN:>5}")

print("-" * 70)


# print(f"\nTotal Time: {round(end_time - start_time, 2)} seconds")

print(f"\n{'Total Time':<20}: {round(end_time - start_time, 2)} seconds\n")