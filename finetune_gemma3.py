import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ======================
# CONFIG
# ======================

BASE_MODEL = "google/gemma-3-270m-it"
DATA_PATH = "gemma_finetune.jsonl"
OUTPUT_DIR = "gemma_fine_tuning/gemma_vito_3"

EPOCHS = 3
BATCH_SIZE = 2
LR = 2e-4


# ======================
# DEVICE
# ======================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# ======================
# LOAD TOKENIZER
# ======================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    token=os.getenv("HF_TOKEN"),
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token


# ======================
# LOAD MODEL (4BIT)
# ======================

print("Loading base model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    token=os.getenv("HF_TOKEN"),
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)


# ======================
# LORA CONFIG
# ======================

print("Configuring LoRA...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


# ======================
# LOAD DATASET
# ======================

print("Loading dataset...")

dataset = load_dataset("json", data_files=DATA_PATH)


# def tokenize(example):
#     return tokenizer(
#         example["text"],
#         truncation=True,
#         padding="max_length",
#         max_length=256,
#     )

# def tokenize(example):

#     text = f"""{example['prompt']} {example['completion']}"""

#     return tokenizer(
#         text,
#         truncation=True,
#         padding="max_length",
#         max_length=256,
#     )

def tokenize(example):

    text = [
        p + " " + c
        for p, c in zip(example["prompt"], example["completion"])
    ]

    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
    )


# dataset = dataset.map(tokenize, batched=True)

dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names,
)


# ======================
# DATA COLLATOR
# ======================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# ======================
# TRAINING ARGS
# ======================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    fp16=False,
    report_to="none",
    optim="paged_adamw_8bit"
)


# ======================
# TRAINER
# ======================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)


# ======================
# TRAIN
# ======================

print("START TRAINING 🚀")

trainer.train()

print("TRAINING FINISHED ✅")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("MODEL SAVED TO:", OUTPUT_DIR)