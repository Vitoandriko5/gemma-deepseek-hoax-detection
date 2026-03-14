import ollama
import json

model_name = "deepseek-llm:7b"

news_text = "The earth is flat and NASA hides the truth."

prompt = f"""
You are a fake news detection system.

Analyze the news below and classify it.

Return your answer ONLY in JSON format like this:
{{
  "label": "Hoax" or "Factual",
  "confidence": "0-100%"
}}

News:
{news_text}
"""

# Call model
response = ollama.chat(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0}
)

raw_output = response["message"]["content"]

print("RAW OUTPUT:")
print(raw_output)

try:
    data = json.loads(raw_output)

    label = data.get("label", "UNKNOWN")
    confidence = data.get("confidence", "0%")

    print("\n===== FINAL RESULT =====")
    print("News      :", news_text)
    print("Label     :", label)
    print("Confidence:", confidence)

except:
    print("\nERROR: Output is not valid JSON")