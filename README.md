# Fake News Classification using LLMs

This project evaluates the performance of Large Language Models (LLMs) for binary fake news classification (Hoax vs. Factual). The experiments compare different open-source models using the same prompt and dataset to ensure a fair evaluation.

The models evaluated in this project include:

- Gemma-3 4B
- DeepSeek-LLM 7B

All experiments were conducted locally using Ollama.

---

# Objective

The goal of this project is to analyze how well different LLMs can detect fake news articles. The evaluation compares model performance across multiple dataset sizes.

Dataset sizes used in the experiments:

- 300 samples
- 1000 samples
- 2000 samples

---

# Dataset

The dataset used in this project contains news articles with the following fields:

- title – news headline
- text – news content
- label – classification label

Label format:

1 = Hoax  
0 = Factual

Due to GitHub file size limitations, the dataset is not included in this repository.

You can download the dataset from the following link:

(https://drive.google.com/drive/folders/1T5ERqfEK5kWnRCYVU_c4PSqMumGxDnqH?usp=sharing)

After downloading, place the dataset file in the project directory:

hf_fake_news.csv

---

# Methodology

Each news article is provided to the model using the following prompt:

You are a strict fake news classifier.

Classify the news into ONE of two categories:

Hoax = contains false claims, misinformation, or unverifiable statements  
Factual = contains verifiable, credible, or neutral information

You MUST choose exactly one label.

The model must answer with only one word:

Hoax  
or  
Factual

All models are evaluated using the same prompt to ensure fair comparison.

---

# Evaluation Metrics

The model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

A confusion matrix is also used to analyze classification errors.

---

# Example Results (300 Samples)

| Model | Accuracy | Precision | Recall | F1-score |
|------|------|------|------|------|
| Gemma-3 4B | 90.33% | 89.83% | 93.53% | 91.64% |
| DeepSeek-LLM 7B | 69.00% | 66.52% | 91.18% | 76.92% |

---

# Output

During the experiment, model predictions are saved into CSV files.

Example output format:

Index | Title | True Label | Predicted Label | Result
----- | ----- | ---------- | --------------- | ------
1 | Example news | Hoax | Factual | WRONG

These files allow detailed error analysis of model predictions.

---

# Tools

This project uses:

- Python
- Ollama
- CSV dataset processing
- Local LLM inference

---

# Author

Natanael Vito Andriko  
Computer Science Student – Binus University
