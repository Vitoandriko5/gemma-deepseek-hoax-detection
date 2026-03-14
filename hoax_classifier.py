import ollama
import time

MODEL_NAME = "gemma3:4b"

def classify_news(news_text):
    prompt = f"""
You are a news classification system.

Classify the following news as:
- Hoax
- Non-Hoax

Answer ONLY one word: Hoax or Non-Hoax.

News:
{news_text}
"""

    start_time = time.time()

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        options={"temperature": 0}
    )

    end_time = time.time()

    prediction = response['message']['content'].strip()

    return prediction, round(end_time - start_time, 2)


news_text = "The earth is flat and NASA hides the truth."

prediction, response_time = classify_news(news_text)

print("News:", news_text)
print("Prediction:", prediction)
print("Response Time:", response_time, "seconds")
