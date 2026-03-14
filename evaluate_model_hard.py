import csv
import ollama
import time

model_name = "gemma3:4b"
dataset_file = "dataset_hard.csv"  

TP = 0
TN = 0
FP = 0
FN = 0

start_time = time.time()

with open(dataset_file, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        text = row["text"]
        true_label = row["label"]

        prompt = f"""
Classify the following news as either 'Hoax' or 'Factual'.
Only answer with one word: Hoax or Factual.

News: {text}
"""

        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )

        raw_prediction = response["message"]["content"].strip()

        # 🔥 Normalisasi output supaya lebih aman
        if "Hoax" in raw_prediction:
            prediction = "Hoax"
        elif "Factual" in raw_prediction:
            prediction = "Factual"
        else:
            prediction = "Unknown"

        print(f"True: {true_label} | Predicted: {prediction}")

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

accuracy = (TP + TN) / total if total != 0 else 0
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print("\n===== FINAL EVALUATION =====")
print(f"Dataset Used: {dataset_file}")
print(f"Total Data: {total}")
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-score: {f1_score:.4f}")

print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-score: {f1_score:.4f} ({f1_score*100:.2f}%)")

print("\nConfusion Matrix:")
print(f"TP (Hoax → Hoax): {TP}")
print(f"TN (Factual → Factual): {TN}")
print(f"FP (Factual → Hoax): {FP}")
print(f"FN (Hoax → Factual): {FN}")

print(f"\nTotal Time: {round(end_time - start_time, 2)} seconds")