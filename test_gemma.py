import ollama

response = ollama.chat(
    model='gemma:2b',
    messages=[
        {
            'role': 'user',
            'content': 'Classify this news as Hoax or Non-Hoax. Answer only one word.\n\nNews: The earth is flat and NASA hides the truth.'
        }
    ]
)

print(response['message']['content'])