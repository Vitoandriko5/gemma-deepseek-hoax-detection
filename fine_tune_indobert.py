import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ========================
# CONFIG
# ========================

MODEL_NAME = "indobenchmark/indobert-base-p1"
DATASET_PATH = "hf_fake_news_indo.csv"
OUTPUT_DIR = "./indobert_fake_news"

MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# ========================
# LOAD DATA
# ========================

print("\nLoading dataset...")
df = pd.read_csv(DATASET_PATH)

df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
df = df[["text", "label"]]

print("\nLabel distribution (FULL DATASET):")
print(df["label"].value_counts())
print("\nLabel distribution (%):")
print(df["label"].value_counts(normalize=True))

# ========================
# TRAIN / VAL SPLIT
# ========================

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    stratify=df["label"]
)

print("\nTrain label distribution:")
print(train_df["label"].value_counts())

print("\nValidation label distribution:")
print(val_df["label"].value_counts())

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ========================
# TOKENIZER
# ========================

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

# ========================
# LOAD MODEL
# ========================

print("\nLoading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(device)

# ========================
# METRICS
# ========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# ========================
# TRAINING ARGUMENTS
# ========================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_dir="./logs",
    load_best_model_at_end=True,
)

# ========================
# TRAINER
# ========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# ========================
# TRAIN
# ========================

print("\nStarting training...")
trainer.train()

print("\nSaving model...")
trainer.save_model(OUTPUT_DIR)

print("\nTraining completed successfully!")