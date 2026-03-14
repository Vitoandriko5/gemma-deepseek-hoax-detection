import csv
import ollama
import time
import random

model_name = "deepseek-llm:7b"
MAX_DATA = 2000   
SLEEP_TIME = 0.5

TP = TN = FP = FN = 0
UNKNOWN = 0

start_time = time.time()

# def normalize_prediction(text):
#     """
#     Paksa output jadi Hoax / Factual / UNKNOWN
#     """
#     t = text.lower()

#     if "hoax" in t:
#         return "Hoax"

#     if "factual" in t:
#         return "Factual"

#     return "UNKNOWN"

# def normalize_prediction(text):
#     """
#     Force output to Hoax or Factual only (no UNKNOWN)
#     """
#     t = text.lower().strip()

#     if "hoax" in t or "fake" in t or "false" in t:
#         return "Hoax"

#     if "factual" in t or "fact" in t or "true" in t or "real" in t:
#         return "Factual"

#     return "Factual"

def normalize_prediction(text):
    """
    Normalize model output to Hoax / Factual only
    (No UNKNOWN, No bias)
    """

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

    return random.choice(["Hoax", "Factual"])


with open("hf_fake_news.csv", newline="", encoding="utf-8") as csvfile:

    reader = list(csv.DictReader(csvfile))

    output_file = open("deepseek_predictions.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(output_file)

    writer.writerow([
        "Index",
        "Title",
        "True_Label",
        "Predicted_Label",
        "Result"
    ])


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

#         prompt = f"""
# You are a professional news fact checker.

# Your task is binary classification.

# Rules:

# - Answer "Hoax" ONLY if there is clear evidence the news is false, misleading, or fabricated.
# - Answer "Factual" if the news appears credible, neutral, or lacks strong evidence of being fake.


# You MUST answer with exactly ONE word:
# Hoax
# or
# Factual

# No explanation.
# No punctuation.

# News:
# {full_text}

# Answer:
# """
        
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

#         prompt = f"""
# You are an expert in detecting fake news.

# Analyze the news carefully.
# If the news contains misleading, false, or unverified information, classify it as Hoax.
# If it is reliable and factual, classify it as Factual.

# Answer ONLY with one word:
# Hoax or Factual

# News:
# {full_text}
# """

        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            # options={
            #     "temperature": 0,
            #     "num_predict": 5
            # }

            options={
                "temperature": 0.3,
                "num_predict": 2,
                "stop": ["\n", ".", ",", " "]
            }

        )

        time.sleep(SLEEP_TIME)

        raw_output = response["message"]["content"].strip()

        prediction = normalize_prediction(raw_output)
        result = "CORRECT" if true_label == prediction else "WRONG"

        writer.writerow([
            i+1,
            title,
            true_label,
            prediction,
            result
        ])


        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (MAX_DATA - (i + 1))

        eta_min = int(remaining // 60)
        eta_sec = int(remaining % 60)

        progress = (i + 1) / MAX_DATA * 100



        print(
    f"[{i+1:>4}/{MAX_DATA:<4}] | "
    f"True: {true_label:<8} | "
    f"Predicted: {prediction:<8} | "
    f"ETA: {eta_min:02d}m {eta_sec:02d}s"
        )

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

    output_file.close()
end_time = time.time()

total = TP + TN + FP + FN

accuracy = (TP + TN) / total if total else 0
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0


# print("\n===== FINAL EVALUATION (DEEPSEEK) =====\n")

# print(f"Model: {model_name}")
# print(f"Total Data (Valid): {total}")
# print(f"Unknown Output    : {UNKNOWN}")


# print(f"\nAccuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
# print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
# print(f"Recall   : {recall:.4f} ({recall*100:.2f}%)")
# print(f"F1-score : {f1:.4f} ({f1*100:.2f}%)")

# print("\nConfusion Matrix:")
# print(f"TP: {TP}")
# print(f"TN: {TN}")
# print(f"FP: {FP}")
# print(f"FN: {FN}")

# print(f"\nTotal Time: {round(end_time - start_time, 2)} seconds")

print("\n===== FINAL EVALUATION (DEEPSEEK) =====\n")

print(f"{'Model':<20}: {model_name}")
print(f"{'Total Data':<20}: {total}")
print(f"{'Unknown Output':<20}: {UNKNOWN}\n")

print(f"{'Accuracy':<20}: {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"{'Precision':<20}: {precision:.4f}  ({precision*100:.2f}%)")
print(f"{'Recall':<20}: {recall:.4f}  ({recall*100:.2f}%)")
print(f"{'F1-score':<20}: {f1:.4f}  ({f1*100:.2f}%)")

# print("\nConfusion Matrix:")
# print(f"{'TP (Hoax news → Hoax detected)':<30}: {TP}")
# print(f"{'TN (Factual news → Factual detected)':<30}: {TN}")
# print(f"{'FP (Factual  news → Hoax detected)':<30}: {FP}")
# print(f"{'FN (Hoax news → Factual detected)':<30}: {FN}")

print("\nConfusion Matrix:")
print("-" * 70)

print(f"{'TP (Hoax news → Hoax predicted)':<45} | {TP:>5}")
print(f"{'TN (Factual news → Factual predicted)':<45} | {TN:>5}")
print(f"{'FP (Factual news → Hoax predicted)':<45} | {FP:>5}")
print(f"{'FN (Hoax news → Factual predicted)':<45} | {FN:>5}")

print("-" * 70)



print(f"\n{'Total Time':<20}: {round(end_time - start_time, 2)} seconds\n")