from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import torch
from datasets import load_dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from datasets import load_dataset
from transformers import *


def relufy_stage1_llama(model):
    """Stage-1: Replace SiLU in LLaMA FFN with ReLU."""
    replaced = 0
    for block in model.model.layers:
        mlp = block.mlp
        if hasattr(mlp, "act_fn"):
            mlp.act_fn = nn.ReLU()
            replaced += 1
    print(f"[Stage-1] LLaMA FFN activations replaced with ReLU in {replaced} blocks.")

from transformers import BitsAndBytesConfig

def build_refinedweb(tokenizer, seq_len: int):
    raw = load_dataset(
        "tiiuae/falcon-refinedweb",
        split="train",
        streaming=True,
    )

    # -------- Tokenizer pass --------
    def tok(batch):
        return tokenizer(
            batch["content"],
            truncation=False,
            add_special_tokens=False,
        )

    tokenized = raw.map(tok, batched=True, remove_columns=raw.column_names)

    # -------- Safe packer: never return empty lists --------
    def pack(batch):
        ids = []

        # collect tokens
        for row in batch["input_ids"]:
            if row:              # skip empty rows
                ids.extend(row)

        # if not enough tokens, return None (NOT empty lists)
        if len(ids) < seq_len:
            return None

        # compute number of full sequences
        n_chunks = len(ids) // seq_len
        if n_chunks == 0:
            return None

        # produce EXACT chunks
        chunks = []
        for i in range(n_chunks):
            start = i * seq_len
            end = start + seq_len
            seq = ids[start:end]
            if len(seq) == seq_len:
                chunks.append(seq)

        # if no valid chunks, return None
        if not chunks:
            return None

        attention_masks = [[1] * seq_len for _ in chunks]

        return {
            "input_ids": chunks,
            "labels": chunks.copy(),
            "attention_mask": attention_masks,
        }

    # map + drop all None entries automatically
    packed = tokenized.map(
        pack,
        batched=True,
        remove_columns=tokenized.column_names,
    )

    # NOW we filter out None (the only safe way)
    packed = packed.filter(lambda x: x is not None)

    # format to torch
    return packed.with_format("torch")



def train_refinedweb(
    model,
    tokenizer,
    seq_len=512,
    batch_size=1,
    grad_accum=8,
    lr=1e-4,
    warmup_ratio=0.01,
    max_tokens=5_000_000,
    log_interval=20,
):

    ds = build_refinedweb(tokenizer, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=default_data_collator)

    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    tokens_per_step = batch_size * seq_len
    approx_steps = max_tokens // (tokens_per_step * grad_accum)
    warmup_steps = max(1, int(approx_steps * warmup_ratio))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(approx_steps, 1),
    )

    total_tokens = 0
    step = 0
    opt_step = 0
    losses = []  # buffer for logging

    print(f"[INFO] approx_steps={approx_steps}, warmup={warmup_steps}")

    for batch in loader:
        step += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        seq_len_in_batch = batch["input_ids"].shape[1]

        # construct explicit position_ids for OPT
        position_ids = torch.arange(
            seq_len_in_batch,
            device=device
        ).unsqueeze(0)

        # -------- Forward pass --------
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(
                **batch,
                position_ids=position_ids,
            )
            loss = outputs.loss / grad_accum
            print(f"loss is {loss}\n")

        # record *unscaled* loss for logging
        losses.append(loss.item())

        # -------- Backward --------
        loss.backward()

        # -------- Accumulation boundary --------
        if step % grad_accum == 0:

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # LR schedule
            scheduler.step()

            # Tracking
            opt_step += 1
            total_tokens += tokens_per_step

            # -------- Safe logging --------
            if opt_step % log_interval == 0:
                if len(losses) > 0:
                    avg = sum(losses) / len(losses)
                else:
                    avg = float('nan')

                print(f"[{opt_step}] tokens={total_tokens/1e6:.2f}M loss={avg:.4f}")
                losses = []  # reset

            # -------- Exit condition --------
            if total_tokens >= max_tokens:
                print(f"[DONE] Hit token budget: {max_tokens}")
                break



class NormWithReLU(nn.Module):
    """Wrap a normalization layer and apply an extra ReLU after it."""
    def __init__(self, norm_layer):
        super().__init__()
        self.norm = norm_layer
        self.relu = nn.ReLU()

    def forward(self, hidden_states):
        return self.relu(self.norm(hidden_states))
def relufy_stage2_llama(model):
    replaced = 0
    for block in model.model.layers:
        # Replace attention input norm
        block.input_layernorm = NormWithReLU(block.input_layernorm)

        # Replace post-attn norm
        block.post_attention_layernorm = NormWithReLU(block.post_attention_layernorm)

"""
LLaMA-2-7B + LoRA fine-tuning on PIQA (ranking-loss objective)
Expected accuracy: ~0.78
"""

import torch, math
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)

# ---------------------------------------------------------------------
# 1. Load PIQA
# ---------------------------------------------------------------------
BASE = "https://yonatanbisk.com/piqa/data"
train = load_dataset("json", data_files={"train": f"{BASE}/train.jsonl"})["train"]
tlabels = load_dataset("text", data_files={"train": f"{BASE}/train-labels.lst"})["train"]
train = train.add_column("label", [int(x["text"]) for x in tlabels])

valid = load_dataset("json", data_files={"validation": f"{BASE}/valid.jsonl"})["validation"]
vlabels = load_dataset("text", data_files={"validation": f"{BASE}/valid-labels.lst"})["validation"]
valid = valid.add_column("label", [int(x["text"]) for x in vlabels])

import torch
from tqdm import tqdm

@torch.no_grad()
def evaluate_piqa(model, tokenizer, dataset, batch_size=1):
    """
    PIQA evaluation for causal LMs.
    Returns accuracy (0.0 - 1.0).
    """
    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = len(dataset)

    for example in tqdm(dataset, desc="Evaluating PIQA"):
        goal = example["goal"].strip()
        sol1 = example["sol1"].strip()
        sol2 = example["sol2"].strip()
        gold = example["label"]

        # Build prompt for each candidate
        texts = [
            f"{goal}\n{sol1}",
            f"{goal}\n{sol2}",
        ]

        scores = []
        for t in texts:
            tokens = tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            tokens = {k: v.to(device) for k, v in tokens.items()}

            # Standard LM loss: lower is better ⇒ score = -loss
            out = model(**tokens, labels=tokens["input_ids"])
            scores.append(-out.loss.item())

        pred = int(scores[1] > scores[0])  # choose higher LL
        correct += (pred == gold)

    acc = correct / total
    print(f"\nPIQA Accuracy: {acc:.4f}")
    return acc

def load_model():
    model_name = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(
    model_name, use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def train():
    model, tokenizer = load_model()
    relufy_stage1_llama(model)
    print(model.model.layers[0].mlp.act_fn)
    train_refinedweb(
        model,
        tokenizer,
        seq_len=1024,
        batch_size=1,
        grad_accum=8,
        lr=1.5e-5,
        max_tokens=6000000,    # adjust for your Colab runtime
        )
    path = "/root/relufied_llama2"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    evaluate_piqa(model.eval(), tokenizer, valid)

if __name__ == "__main__":
    train()
