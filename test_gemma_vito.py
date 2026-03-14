# import os
# import torch
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     BitsAndBytesConfig
# )
# from peft import PeftModel

# # CONFIG

# BASE_MODEL = "google/gemma-2b"
# LORA_PATH = "../gemma_fine_tuning/gemma_vito/checkpoint-750"

# HF_TOKEN = os.getenv("HF_TOKEN")  # ambil dari env


# # DEVICE

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", device)


# # QUANT CONFIG

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )



# # LOAD TOKENIZER

# print("Loading tokenizer...")

# tokenizer = AutoTokenizer.from_pretrained(
#     BASE_MODEL,
#     token=HF_TOKEN,
#     trust_remote_code=True
# )


# # LOAD BASE MODEL

# print("Loading base model...")

# model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     token=HF_TOKEN,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True
# )


# # LOAD LORA

# print("Loading LoRA...")

# model = PeftModel.from_pretrained(model, LORA_PATH)

# model.eval()


# # ASK FUNCTION

# def ask(text):

#     prompt = f"""
# Anda adalah sistem klasifikasi berita.

# Tugas Anda:
# Jawab hanya dengan satu kata: Hoax atau Fakta.

# Berita:
# {text}

# Jawaban:
# """

#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt"
#     ).to(model.device)

#     # out = model.generate(
#     #     **inputs,
#     #     max_new_tokens=100,
#     #     temperature=0.2
#     # )

#     out = model.generate(
#     **inputs,
#     max_new_tokens=10,
#     do_sample=False,
#     eos_token_id=tokenizer.eos_token_id
#     )


#     decoded = tokenizer.decode(
#         out[0],
#         skip_special_tokens=True
#     )

#     result = decoded.split("Jawaban:")[-1].strip()

#     print("=" * 50)
#     print("BERITA:")
#     print(text)
#     print("\nHASIL:")
#     print(result)
#     print("=" * 50)

# # TEST

# ask("Vaksin menyebabkan manusia jadi zombie")

# ask("Pemerintah Indonesia menetapkan 17 Agustus sebagai hari libur nasional")

# ask("Minum air garam bisa menyembuhkan semua penyakit")

# ask("BMKG memprediksi hujan lebat di Jakarta minggu depan")

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel


# =========================
# CONFIG
# =========================

BASE_MODEL = "google/gemma-2b"
LORA_PATH = "../gemma_fine_tuning/gemma_vito/checkpoint-750"

HF_TOKEN = os.getenv("HF_TOKEN")


# =========================
# DEVICE
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# =========================
# QUANT CONFIG
# =========================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# =========================
# LOAD TOKENIZER
# =========================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    trust_remote_code=True
)


# =========================
# LOAD BASE MODEL
# =========================

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)


# =========================
# LOAD LORA
# =========================

print("Loading LoRA...")

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()


# =========================
# ASK FUNCTION
# =========================

def ask(text):

    prompt = f"""
Anda adalah sistem klasifikasi berita.

Tugas Anda:
1. Analisis apakah berita ini masuk akal.
2. Gunakan pengetahuan umum.
3. Tentukan Hoax atau Fakta.

Berita:
{text}

Jawaban (Hoax/Fakta):
"""

#     prompt = f"""
# Anda adalah sistem klasifikasi berita.

# Tugas Anda:
# Jawab hanya dengan satu kata: Hoax atau Fakta.

# Berita:
# {text}

# Jawaban:
# """

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(
        out[0],
        skip_special_tokens=True
    ).lower()


    # =========================
    # CLEAN OUTPUT
    # =========================

    if "hoax" in decoded:
        result = "Hoax"
    elif "fakta" in decoded:
        result = "Fakta"
    else:
        result = "Tidak terdeteksi"


    print("=" * 50)
    print("BERITA:")
    print(text)
    print("\nHASIL:")
    print(result)
    print("=" * 50)



# =========================
# TEST
# =========================

ask("Vaksin menyebabkan manusia jadi zombie")

ask("Pemerintah Indonesia menetapkan 17 Agustus sebagai hari libur nasional")
