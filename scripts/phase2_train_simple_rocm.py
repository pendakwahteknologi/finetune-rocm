#!/usr/bin/env python3
"""
LoRA fine-tuning for AMD ROCm.
Manual training loop optimized for stability on gfx1201 (RDNA 4).
"""

import gc
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# =============================================================================
# Configuration
# =============================================================================
BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_HUB_TOKEN", "")

BASE_DIR = Path(os.environ.get("BASE_DIR", str(Path.home() / "finetune-rocm")))
DATA_DIR = BASE_DIR / "data"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(BASE_DIR / "models" / "lora_simple_rocm_out")))
LOG_DIR = BASE_DIR / "results"

CHAT_JSONL = DATA_DIR / "train_chatml.jsonl"

# Training parameters
MAX_TRAIN_ROWS = int(os.environ.get("MAX_TRAIN_ROWS", "2000"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
LR = float(os.environ.get("LR", "2e-4"))
USE_FP16 = os.environ.get("USE_FP16", "1") == "1"

# LoRA parameters
LORA_R = int(os.environ.get("LORA_R", "8"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))

# Logging
LOG_EVERY = int(os.environ.get("LOG_EVERY", "10"))

# ROCm environment defaults
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.1")
os.environ.setdefault("GPU_MAX_HW_QUEUES", "2")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")


def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        return torch.device("cuda")
    print("[warn] GPU unavailable or disabled - falling back to CPU")
    return torch.device("cpu")


def gpu_mem():
    if not torch.cuda.is_available():
        return None
    return (
        torch.cuda.memory_allocated(0),
        torch.cuda.memory_reserved(0),
        torch.cuda.get_device_properties(0).total_memory,
    )


def fmt_bytes(b):
    return f"{b / 1024**3:.2f} GiB"


def print_gpu_info():
    if not torch.cuda.is_available():
        return
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if hasattr(props, "gcnArchName"):
        print(f"Arch: {props.gcnArchName}")
    print(f"Total VRAM: {fmt_bytes(props.total_memory)}")


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(dtype):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        device_map=None,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    model.config.use_cache = False
    return model


def collate_dynamic_pad(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for x in batch:
        ids = x["input_ids"]
        lab = x["labels"]
        pad_n = max_len - len(ids)

        input_ids.append(ids + [pad_id] * pad_n)
        attention_mask.append([1] * len(ids) + [0] * pad_n)
        labels.append(lab + [-100] * pad_n)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ensure_dirs()
    start = datetime.now()
    print(f"Start: {start:%Y-%m-%d %H:%M:%S}")

    device = get_device()
    dtype = torch.float16 if USE_FP16 else torch.float32
    print_gpu_info()
    print(f"Base model: {BASE_MODEL}")
    print(f"Data file: {CHAT_JSONL}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Seq len: {SEQ_LEN}, batch: {BATCH_SIZE}, grad_accum: {GRAD_ACCUM}")

    # Tokenizer
    tokenizer = load_tokenizer()

    # Dataset
    ds = load_dataset("json", data_files=str(CHAT_JSONL), split="train")
    if MAX_TRAIN_ROWS > 0 and len(ds) > MAX_TRAIN_ROWS:
        ds = ds.select(range(MAX_TRAIN_ROWS))
    print(f"Dataset rows selected: {len(ds)}")

    texts = []
    for ex in ds:
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    tokenized = []
    lengths = []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=SEQ_LEN, padding=False)
        ids = enc["input_ids"]
        lengths.append(len(ids))
        tokenized.append({"input_ids": ids, "labels": ids.copy()})

    print(f"Avg tokens: {statistics.mean(lengths):.0f}, max: {max(lengths)}")

    loader = DataLoader(
        tokenized,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=lambda b: collate_dynamic_pad(b, tokenizer.pad_token_id),
    )
    print(f"Batches per epoch: {len(loader)}")

    # Model
    model = load_model(dtype)
    model.to(device)

    # LoRA
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    model = get_peft_model(model, lora_config)
    model.train()
    print("LoRA adapters applied")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        foreach=False,
    )

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * EPOCHS
    print(f"Total training steps: {total_steps}")

    start_time = time.time()
    global_step = 0
    loss_window = []

    for epoch in range(1, EPOCHS + 1):
        for batch in loader:
            global_step += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss_scalar = float(loss.detach().cpu().item())
            loss_window.append(loss_scalar)
            if len(loss_window) > LOG_EVERY:
                loss_window.pop(0)

            (loss / GRAD_ACCUM).backward()

            if global_step % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                avg_loss = statistics.mean(loss_window)
                sps = (global_step * BATCH_SIZE) / max(elapsed, 1e-9)
                print(
                    f"step={global_step}/{total_steps} "
                    f"epoch={epoch}/{EPOCHS} "
                    f"loss={avg_loss:.4f} "
                    f"samples_per_sec={sps:.2f}"
                )

    model_cpu = model.to("cpu")
    cleanup()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_cpu.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.1f}s")
    print(f"Saved adapter: {OUT_DIR}")


if __name__ == "__main__":
    main()
