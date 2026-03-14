import csv
import ollama
import time

model_name = "gemma3:4b"

correct = 0
total = 0

start_total_time = time.time()

with open("dataset.csv", newline="", encoding="utf-8") as csvfile:
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

        prediction = response["message"]["content"].strip()

        if prediction == true_label:
            correct += 1

        total += 1

        print(f"True: {true_label} | Predicted: {prediction}")

end_total_time = time.time()

accuracy = (correct / total) * 100

print("\n==========================")
print(f"Total Data: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Total Evaluation Time: {round(end_total_time - start_total_time, 2)} seconds")