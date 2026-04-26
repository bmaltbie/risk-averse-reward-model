#!/usr/bin/env python3
"""
rft_pipeline.py — Reward-Model Tuning checklist from Ben.pdf (items 1-14).

Two phases:
  1) Validation ladder: train at 2-3 LRs near --middle_lr (extended ladder
     [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]; starting default 1e-5),
     eval on `reward_model_validation`, pick best LR by
     pairwise_accuracy_ties_half_credit.
  2) Held-out: train at best LR with seeds 1, 2, 3; eval each seed on
     `reward_model_high_stakes_test`, `reward_model_astronomical_stakes_deployment`,
     `reward_model_steals_test`.

Parameterized by --base_model so the same script runs Qwen3-1.7B / 8B / 14B
(Ben.pdf items 13, 14). Architecture matches sweep.py / colab: AutoModel
backbone (fp16) + LoRA (FEATURE_EXTRACTION) + separately trained
nn.Linear(hidden, 1) reward head (fp32), saved as reward_head.pt alongside
the LoRA adapter.

Per-run eval writes two JSONs per held-out dataset under each run dir:
  - eval_<alias>.json     (in-process, reuses the in-memory backbone+head)
  - eval_rm_<alias>.json  (subprocess: evaluate_reward_model.py from the
                           submodule, which auto-loads reward_head.pt
                           alongside the LoRA adapter at --model_path).
The two paths are kept in parallel so their pairwise metrics can be
cross-checked. Disable the script eval with --skip_script_eval.

Resumable: per-run `complete.json` and per-dataset `eval_<alias>.json`
(plus `eval_rm_<alias>.json`) sentinels. `status.json` is rewritten after
every run.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Heavy ML imports are lazy — --help and --dry_run don't need torch/transformers/peft.
# See _import_ml() below.

# ============================================================
# PARAMETERS
# ============================================================
# All tunable defaults live in this block. Parameters are grouped by role.
# Each entry below the "---" dividers carries an inline comment naming the
# CLI flag that overrides it (if any). Entries marked "not flag-overridable"
# can only be changed by editing this file.

# --- Paths (not flag-overridable) ---
PROJECT_ROOT = Path(__file__).resolve().parent
SUBMODULE_ROOT = PROJECT_ROOT / "eval" / "risk-averse-ai-eval"

# --- Training CSV (override: --train_csv) ---
DEFAULT_TRAIN_CSV = SUBMODULE_ROOT / "data" / "2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv"

# --- Base model (override: --base_model) ---
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"

# --- LR ladder (not flag-overridable) ---
# Fixed ladder of allowed learning-rate rungs. Any LR the pipeline uses
# (middle_lr, best_lr) must snap to one of these. Ben.pdf line 628 prescribes
# the first five; 5e-4 and 1e-3 are our extensions.
LR_LADDER: List[float] = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

# --- Starting / best LR (override: --middle_lr, --best_lr) ---
DEFAULT_MIDDLE_LR: float = 1e-5

# --- LoRA config (not flag-overridable; Ben.pdf line 632 locks these) ---
# task_type is FEATURE_EXTRACTION to match sweep.py / colab architecture:
# AutoModel backbone + separately trained nn.Linear(hidden, 1) reward head
# (saved alongside the LoRA adapter as reward_head.pt).
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- Reward head init (not flag-overridable; matches sweep.py:91) ---
REWARD_HEAD_INIT_STD = 0.01

# --- Training hyperparameters ---
DEFAULT_EPOCHS = 5                  # override: --epochs               (ablation values: 3, 5, 7)
DEFAULT_BATCH_SIZE = 2              # override: --batch_size           (Ben.pdf blank)
DEFAULT_GRAD_ACCUM_STEPS = 32       # override: --grad_accum_steps     (Ben.pdf blank)
DEFAULT_MAX_LENGTH = 1024           # override: --max_length
DEFAULT_WEIGHT_DECAY = 0.05         # override: --weight_decay         (Ben.pdf blank; matches sweep.py)
DEFAULT_WARMUP_RATIO = 0.1          # override: --warmup_ratio         (Ben.pdf line 661)
DEFAULT_MAX_GRAD_NORM = 1.0         # override: --max_grad_norm
DEFAULT_TORCH_DTYPE = "float16"     # override: --torch_dtype (backbone only; head is fp32)
DEFAULT_GRAD_CKPT = True            # override: --grad_ckpt / --no_grad_ckpt

# --- Eval parameters ---
DEFAULT_VAL_NUM_PAIRS = 200                 # override: --val_num_pairs
DEFAULT_HELDOUT_NUM_PAIRS = 1000            # override: --heldout_num_pairs
DEFAULT_EVAL_BATCH_SIZE = 16                # override: --eval_batch_size
DEFAULT_EVAL_MAX_LENGTH = 4096              # override: --eval_max_length
DEFAULT_HELDOUT_DATASETS = [                # override: --heldout_datasets
    "reward_model_high_stakes_test",
    "reward_model_astronomical_stakes_deployment",
    "reward_model_steals_test",
]

# --- Dataset alias → submodule CSV path (not flag-overridable) ---
# Mirrors DATASET_ALIASES in eval/risk-averse-ai-eval/evaluate_reward_model.py:55-66
# so in-process eval reads the same files the subprocess would have resolved.
DATASET_ALIAS_PATHS = {
    "reward_model_validation":
        "data/2026_03_22_reward_model_val_set_400_Rebels_clean.csv",
    "reward_model_high_stakes_test":
        "data/2026_03_22_high_stakes_test_set_746_Rebels_CoTs_for_evaluating_reward_model_from_Sonnet.csv",
    "reward_model_astronomical_stakes_deployment":
        "data/2026_03_22_astronomical_stakes_deployment_set_707_Rebels_CoTs_for_evaluating_reward_model_from_Sonnet.csv",
    "reward_model_steals_test":
        "data/2026_03_22_test_set_928_Steals_CoTs_for_evaluating_reward_model_from_Sonnet.csv",
}

# --- Seeds ---
DEFAULT_SEEDS_VALIDATION = [1]              # override: --seeds_validation  (Ben.pdf line 58)
DEFAULT_SEEDS_HELDOUT = [1, 2, 3]           # override: --seeds_heldout     (Ben.pdf line 62)

# --- System prompt ---
# We do NOT use a system prompt during training or eval. Rationale:
# (1) sweep.py and colab_notebook.ipynb also use no system prompt and produce
#     our best measured results.
# (2) Since option-2 in-process eval controls both train and eval formatting,
#     there is no external parity to maintain with evaluate_reward_model.py.
# (3) DEFAULT_SYSTEM_PROMPT was ~253 tokens; dropping it frees significant
#     max_length budget for long CoT responses.
# Chat template still applies; it just collapses to [user, assistant] turns.

# ============================================================


class OOMSkipRun(RuntimeError):
    """Raised when a CUDA OOM should abort the current (phase, lr, seed) and continue."""


def _import_ml():
    """Lazy import of torch/transformers/peft. Returns a namespace-like dict."""
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    from torch.utils.data import DataLoader as _DataLoader, Dataset as _Dataset
    from transformers import (
        AutoModel as _AutoModel,
        AutoTokenizer as _ATok,
        get_cosine_schedule_with_warmup as _cosine,
    )
    from peft import LoraConfig as _LoraConfig, PeftModel as _PeftModel, TaskType as _TaskType, get_peft_model as _get_peft_model
    return {
        "torch": _torch, "nn": _nn, "F": _F,
        "DataLoader": _DataLoader, "Dataset": _Dataset,
        "AutoModel": _AutoModel,
        "AutoTokenizer": _ATok,
        "get_cosine_schedule_with_warmup": _cosine,
        "LoraConfig": _LoraConfig, "PeftModel": _PeftModel, "TaskType": _TaskType,
        "get_peft_model": _get_peft_model,
    }


# --- Shared formatting helpers (must match evaluate_reward_model.py:105-120, 281-310) ---

_INSTRUCTION_PATTERNS = [
    r"\s*You can think before answering,.*?would select\.",
    r"\s*You can think.*?must finish with.*?\.",
]


def remove_instruction_suffix(prompt: str) -> str:
    """Strip old evaluation-instruction text that was embedded in some prompt CSVs."""
    out = prompt
    for pattern in _INSTRUCTION_PATTERNS:
        out = re.sub(pattern, "", out, flags=re.IGNORECASE | re.DOTALL)
    return out.strip()


def format_chat_text(tokenizer, system_prompt: str, prompt: str, response: str) -> str:
    """Format a (system?, user, assistant) turn with chat template. Shared by train and eval."""
    messages: List[Dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def _hf_token() -> Optional[str]:
    """HuggingFace access token for gated models (e.g., meta-llama/*).
    Reads HF_TOKEN from env; returns None if unset (public models work without it)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def parse_torch_dtype_name(name: str):
    """Standalone dtype parser that doesn't require ml dict (imports torch lazily)."""
    import torch as _torch
    name = name.lower().strip()
    if name in ("bfloat16", "bf16"): return _torch.bfloat16
    if name in ("float16", "fp16", "half"): return _torch.float16
    if name in ("float32", "fp32"): return _torch.float32
    if name == "auto":
        return _torch.bfloat16 if _torch.cuda.is_available() else _torch.float32
    raise ValueError(f"Unknown --torch_dtype {name!r}")


def make_compute_rewards(backbone, reward_head):
    """Build the forward closure shared by training and in-process eval.

    backbone (fp16, PEFT-wrapped) → last non-pad hidden state → fp32 cast
    (no .detach(); LoRA gradients flow) → reward head (fp32) → scalar.
    """
    import torch as _torch

    def compute_rewards(input_ids, attention_mask):
        out = backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = out.last_hidden_state  # (B, T, H) fp16
        seq_lens = attention_mask.sum(dim=1) - 1
        B = input_ids.shape[0]
        last = hidden[_torch.arange(B, device=hidden.device), seq_lens].float()
        return reward_head(last).squeeze(-1)

    return compute_rewards


# ============================================================
# Utilities
# ============================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def lr_tag(lr: float) -> str:
    """Stable filename tag for an LR value (e.g. 1e-05)."""
    return f"{lr:.0e}"


def snap_to_ladder(lr: float, tol: float = 1e-9) -> Optional[float]:
    for rung in LR_LADDER:
        if abs(lr - rung) < tol * max(1.0, rung):
            return rung
    return None


def select_candidate_lrs(middle_lr: float) -> List[float]:
    snapped = snap_to_ladder(middle_lr)
    if snapped is None:
        raise ValueError(
            f"--middle_lr={middle_lr} is not on Ben.pdf ladder {LR_LADDER}. "
            f"Off-ladder values require Elliott approval (Ben.pdf lines 628-636)."
        )
    i = LR_LADDER.index(snapped)
    rungs: List[float] = []
    if i > 0:
        rungs.append(LR_LADDER[i - 1])
    rungs.append(LR_LADDER[i])
    if i < len(LR_LADDER) - 1:
        rungs.append(LR_LADDER[i + 1])
    return rungs


def atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False))
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _try_empty_cache() -> None:
    """Best-effort CUDA cache flush. Does not require torch at module-import time."""
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass


# ============================================================
# Training
# ============================================================

def train_one_run(
    base_model: str,
    lr: float,
    seed: int,
    train_df: pd.DataFrame,
    run_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train a single (lr, seed) run and return (training_summary, model_objs).

    model_objs contains {backbone, reward_head, tokenizer, sys_prompt, device,
    compute_rewards} for immediate in-process eval. Caller is responsible for
    deleting these when done.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoint"

    ml = _import_ml()
    torch = ml["torch"]
    F = ml["F"]
    DataLoader = ml["DataLoader"]
    Dataset = ml["Dataset"]

    def set_seeds():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seeds()

    class PairwiseChatDataset(Dataset):
        def __init__(self, df, tokenizer, system_prompt, max_length):
            required = {"prompt_text", "chosen_full", "rejected_full"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Training CSV missing required columns: {missing}")
            self.df = df.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.system_prompt = system_prompt
            self.max_length = max_length

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            prompt = remove_instruction_suffix(str(row["prompt_text"]))
            c_text = format_chat_text(self.tokenizer, self.system_prompt, prompt, str(row["chosen_full"]))
            r_text = format_chat_text(self.tokenizer, self.system_prompt, prompt, str(row["rejected_full"]))
            c_enc = self.tokenizer(c_text, truncation=True, max_length=self.max_length, add_special_tokens=False)
            r_enc = self.tokenizer(r_text, truncation=True, max_length=self.max_length, add_special_tokens=False)
            return {
                "chosen_input_ids": c_enc["input_ids"],
                "chosen_attention_mask": c_enc["attention_mask"],
                "rejected_input_ids": r_enc["input_ids"],
                "rejected_attention_mask": r_enc["attention_mask"],
            }

    def pad_right(seqs, pad_id):
        max_len = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return ids, mask

    def collate(batch):
        c_ids, c_mask = pad_right([b["chosen_input_ids"] for b in batch], tokenizer.pad_token_id)
        r_ids, r_mask = pad_right([b["rejected_input_ids"] for b in batch], tokenizer.pad_token_id)
        return {
            "chosen": {"input_ids": c_ids, "attention_mask": c_mask},
            "rejected": {"input_ids": r_ids, "attention_mask": r_mask},
        }

    print(f"[train] base_model={base_model} lr={lr:.1e} seed={seed}")
    print(f"[train] run_dir={run_dir}")

    tokenizer = ml["AutoTokenizer"].from_pretrained(
        base_model, trust_remote_code=args.trust_remote_code, token=_hf_token()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    tokenizer.padding_side = "right"

    dtype = parse_torch_dtype_name(args.torch_dtype)
    # FEATURE_EXTRACTION architecture (matches sweep.py:603-675):
    # - AutoModel backbone in fp16 for memory efficiency
    # - Separate nn.Linear(hidden, 1) reward head in fp32 for numerical stability
    # - Hidden states cast fp16 -> fp32 before the head (no .detach(), so
    #   LoRA gradients flow through the backbone normally).
    backbone = ml["AutoModel"].from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if torch.cuda.is_available() else None,
        token=_hf_token(),
    )
    if getattr(backbone.config, "pad_token_id", None) is None:
        backbone.config.pad_token_id = tokenizer.pad_token_id

    if args.grad_ckpt:
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()

    lora_cfg = ml["LoraConfig"](
        task_type=ml["TaskType"].FEATURE_EXTRACTION,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        target_modules=LORA_TARGET_MODULES,
    )
    backbone = ml["get_peft_model"](backbone, lora_cfg)
    backbone.print_trainable_parameters()

    hidden_size = backbone.config.hidden_size
    reward_head = ml["nn"].Linear(hidden_size, 1, bias=True)
    ml["nn"].init.normal_(reward_head.weight, mean=0.0, std=REWARD_HEAD_INIT_STD)
    ml["nn"].init.zeros_(reward_head.bias)
    device = next(backbone.parameters()).device
    reward_head = reward_head.to(device).float()  # fp32 head

    compute_rewards = make_compute_rewards(backbone, reward_head)

    sys_prompt = ""  # no system prompt by design (see module-level comment)

    dataset = PairwiseChatDataset(
        df=train_df,
        tokenizer=tokenizer,
        system_prompt=sys_prompt,
        max_length=args.max_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    trainable = [p for p in backbone.parameters() if p.requires_grad] + list(reward_head.parameters())
    optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    opt_steps_per_epoch = math.ceil(steps_per_epoch / args.grad_accum_steps)
    total_opt_steps = opt_steps_per_epoch * args.epochs
    scheduler = ml["get_cosine_schedule_with_warmup"](
        optim,
        num_warmup_steps=max(1, int(args.warmup_ratio * total_opt_steps)),
        num_training_steps=total_opt_steps,
    )

    history: List[Dict[str, Any]] = []
    start = time.time()

    # Per-epoch validation + best-checkpoint selection.
    # We eval the in-memory model at end of each epoch on reward_model_validation
    # (first --val_num_pairs rows). On every improvement in
    # pairwise_accuracy_ties_half_credit we:
    #   (a) save checkpoint (LoRA adapter + reward_head.pt) to ckpt_dir, and
    #   (b) promote the val eval JSON to eval_reward_model_validation.json
    #       so run_single's resume branch finds it without re-scoring.
    val_scratch = run_dir / ".val_scratch.json"
    val_official = run_dir / "eval_reward_model_validation.json"
    best_val_acc = -math.inf
    best_epoch = 0

    def _save_best_checkpoint():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        backbone.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        torch.save(
            {"reward_head_state_dict": reward_head.state_dict(), "hidden_size": hidden_size},
            ckpt_dir / "reward_head.pt",
        )

    for epoch in range(args.epochs):
        backbone.train()
        reward_head.train()
        epoch_loss_sum = 0.0
        epoch_loss_n = 0
        optim.zero_grad()
        for step, batch in enumerate(loader):
            try:
                chosen = {k: v.to(device, non_blocking=True) for k, v in batch["chosen"].items()}
                rejected = {k: v.to(device, non_blocking=True) for k, v in batch["rejected"].items()}
                r_c = compute_rewards(chosen["input_ids"], chosen["attention_mask"])
                r_r = compute_rewards(rejected["input_ids"], rejected["attention_mask"])
                loss = -F.logsigmoid(r_c - r_r).mean()
                (loss / args.grad_accum_steps).backward()
                epoch_loss_sum += loss.item() * r_c.shape[0]
                epoch_loss_n += r_c.shape[0]
                if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == steps_per_epoch:
                    torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                    optim.step()
                    scheduler.step()
                    optim.zero_grad()
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                msg = str(e).lower()
                if "out of memory" in msg or isinstance(e, torch.cuda.OutOfMemoryError):
                    print(f"[train] WARNING: CUDA OOM at epoch {epoch+1} step {step+1}. Aborting run.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise OOMSkipRun(f"OOM at epoch {epoch+1} step {step+1}") from e
                raise

        avg_loss = epoch_loss_sum / max(epoch_loss_n, 1)
        entry: Dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "lr": scheduler.get_last_lr()[0],
            "wall_seconds": round(time.time() - start, 1),
        }

        # --- per-epoch validation on reward_model_validation ---
        val_model_objs = {
            "backbone": backbone,
            "reward_head": reward_head,
            "tokenizer": tokenizer,
            "sys_prompt": sys_prompt,
            "device": device,
            "compute_rewards": compute_rewards,
        }
        val_result = score_pairs_in_process(
            model_objs=val_model_objs,
            dataset_alias="reward_model_validation",
            num_pairs=args.val_num_pairs,
            output_json=val_scratch,
            args=args,
            tag=f"epoch_{epoch+1}_val",
        )
        val_acc = val_result["pairwise_accuracy_ties_half_credit"]
        entry["val_pairwise_accuracy"] = val_result["pairwise_accuracy"]
        entry["val_pairwise_accuracy_ties_half_credit"] = val_acc
        entry["val_mean_score_margin"] = val_result["mean_score_margin"]
        entry["val_num_pairs"] = val_result["num_pairs"]
        history.append(entry)

        improved = val_acc > best_val_acc
        marker = " * NEW BEST" if improved else ""
        print(f"[train] epoch {epoch+1}/{args.epochs} train_loss={avg_loss:.4f} "
              f"val_acc={val_acc:.4f}{marker}")

        if improved:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            _save_best_checkpoint()
            shutil.copy(val_scratch, val_official)

    if val_scratch.exists():
        val_scratch.unlink()

    # If the last epoch wasn't the best, the in-memory model has drifted from
    # the saved checkpoint. Reload so returned model_objs match disk (which is
    # what the downstream held-out eval will score).
    last_epoch_run = len(history)
    if best_epoch and best_epoch < last_epoch_run:
        print(f"[train] best epoch was {best_epoch} (val_acc={best_val_acc:.4f}); "
              f"reloading best checkpoint into memory (last was epoch {last_epoch_run})")
        del backbone, reward_head, optim, scheduler, loader, dataset
        gc.collect()
        _try_empty_cache()
        reloaded = load_checkpoint_for_eval(base_model, ckpt_dir, args)
        backbone = reloaded["backbone"]
        reward_head = reloaded["reward_head"]
        tokenizer = reloaded["tokenizer"]
        device = reloaded["device"]
        sys_prompt = reloaded["sys_prompt"]
        compute_rewards = reloaded["compute_rewards"]
    else:
        # Free training-only state; keep model/tokenizer for in-process eval.
        del optim, scheduler, loader, dataset

    saved_files = sorted(os.listdir(ckpt_dir)) if ckpt_dir.exists() else []
    has_adapter = any("adapter" in f for f in saved_files)
    has_reward_head = (ckpt_dir / "reward_head.pt").exists()
    if not has_adapter:
        print(f"[train] WARNING: no LoRA adapter file in {ckpt_dir}. Files: {saved_files}")
    if not has_reward_head:
        print(f"[train] WARNING: reward_head.pt missing from {ckpt_dir}")

    atomic_write_json(run_dir / "training_history.json", history)
    summary = {
        "finished_at": now_iso(),
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "best_epoch": best_epoch,
        "best_val_pairwise_accuracy_ties_half_credit": best_val_acc if best_epoch else None,
        "epochs_run": len(history),
        "wall_seconds": round(time.time() - start, 1),
        "checkpoint_files": saved_files,
        "has_adapter": has_adapter,
        "has_reward_head": has_reward_head,
    }
    atomic_write_json(run_dir / "complete.json", summary)

    model_objs = {
        "backbone": backbone,
        "reward_head": reward_head,
        "tokenizer": tokenizer,
        "sys_prompt": sys_prompt,
        "device": device,
        "compute_rewards": compute_rewards,
    }
    return summary, model_objs


# ============================================================
# In-process eval (replaces evaluate_reward_model.py subprocess)
#
# Why in-process: our checkpoints save AutoModel+LoRA+reward_head.pt, but
# evaluate_reward_model.py loads via AutoModelForSequenceClassification and
# does not read reward_head.pt, so subprocess metrics would be garbage.
# Reusing the training-time backbone/reward_head objects here guarantees
# train/eval parity.
# ============================================================

def load_checkpoint_for_eval(base_model: str, ckpt_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Rebuild backbone + reward_head from disk for eval-only runs (resume path)."""
    ml = _import_ml()
    torch = ml["torch"]

    tokenizer = ml["AutoTokenizer"].from_pretrained(
        str(ckpt_dir), trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    tokenizer.padding_side = "right"

    dtype = parse_torch_dtype_name(args.torch_dtype)
    backbone_base = ml["AutoModel"].from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if torch.cuda.is_available() else None,
        token=_hf_token(),
    )
    if getattr(backbone_base.config, "pad_token_id", None) is None:
        backbone_base.config.pad_token_id = tokenizer.pad_token_id
    backbone = ml["PeftModel"].from_pretrained(backbone_base, str(ckpt_dir))

    head_ckpt = torch.load(ckpt_dir / "reward_head.pt", map_location="cpu")
    hidden_size = head_ckpt.get("hidden_size", backbone.config.hidden_size)
    reward_head = ml["nn"].Linear(hidden_size, 1, bias=True)
    reward_head.load_state_dict(head_ckpt["reward_head_state_dict"])
    device = next(backbone.parameters()).device
    reward_head = reward_head.to(device).float()

    return {
        "backbone": backbone,
        "reward_head": reward_head,
        "tokenizer": tokenizer,
        "sys_prompt": "",  # no system prompt by design (see module-level comment)
        "device": device,
        "compute_rewards": make_compute_rewards(backbone, reward_head),
    }


def _logistic_loss(margin: float) -> float:
    """softplus(-margin); numerically stable pairwise log loss."""
    if margin >= 0:
        return math.log1p(math.exp(-margin))
    return -margin + math.log1p(math.exp(margin))


def score_pairs_in_process(
    model_objs: Dict[str, Any],
    dataset_alias: str,
    num_pairs: int,
    output_json: Path,
    args: argparse.Namespace,
    tag: str,
) -> Dict[str, Any]:
    """Bradley-Terry pairwise scoring using the in-memory backbone+reward_head.

    Writes output_json with the same metric keys evaluate_reward_model.py
    emits, so downstream status.json parsing is unchanged.
    """
    import torch as _torch

    backbone = model_objs["backbone"]
    reward_head = model_objs["reward_head"]
    tokenizer = model_objs["tokenizer"]
    sys_prompt = model_objs["sys_prompt"]
    device = model_objs["device"]
    compute_rewards = model_objs["compute_rewards"]

    rel = DATASET_ALIAS_PATHS.get(dataset_alias)
    if rel is None:
        raise ValueError(f"Unknown dataset alias {dataset_alias!r}. Known: {list(DATASET_ALIAS_PATHS)}")
    csv_path = SUBMODULE_ROOT / rel
    if not csv_path.exists():
        raise FileNotFoundError(f"Eval CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = {"prompt_text", "chosen_full", "rejected_full"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    # Dedup exact (prompt, chosen, rejected) triples and cap at num_pairs.
    seen: set = set()
    pairs: List[Dict[str, Any]] = []
    skipped_duplicates = 0
    for _, row in df.iterrows():
        key = (str(row["prompt_text"]), str(row["chosen_full"]), str(row["rejected_full"]))
        if key in seen:
            skipped_duplicates += 1
            continue
        seen.add(key)
        pairs.append({
            "prompt": remove_instruction_suffix(str(row["prompt_text"])),
            "chosen": str(row["chosen_full"]),
            "rejected": str(row["rejected_full"]),
        })
        if num_pairs and len(pairs) >= num_pairs:
            break

    n = len(pairs)
    print(f"[eval:{tag}] scoring {n} pairs from {csv_path.name} (batch_size={args.eval_batch_size}, max_len={args.eval_max_length})")

    def score_batch(batch_pairs: List[Dict[str, Any]], bs: int) -> Tuple[List[float], List[float], List[bool], List[bool]]:
        """Score one batch of pairs. bs is informational."""
        chosen_texts = [format_chat_text(tokenizer, sys_prompt, p["prompt"], p["chosen"]) for p in batch_pairs]
        rejected_texts = [format_chat_text(tokenizer, sys_prompt, p["prompt"], p["rejected"]) for p in batch_pairs]

        # Raw lengths (pre-truncation) for truncation-rate metric.
        raw_c = tokenizer(chosen_texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        raw_r = tokenizer(rejected_texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        trunc_c = [len(ids) > args.eval_max_length for ids in raw_c]
        trunc_r = [len(ids) > args.eval_max_length for ids in raw_r]

        enc_c = tokenizer(chosen_texts, padding=True, truncation=True,
                          max_length=args.eval_max_length, add_special_tokens=False, return_tensors="pt")
        enc_r = tokenizer(rejected_texts, padding=True, truncation=True,
                          max_length=args.eval_max_length, add_special_tokens=False, return_tensors="pt")
        enc_c = {k: v.to(device) for k, v in enc_c.items()}
        enc_r = {k: v.to(device) for k, v in enc_r.items()}

        r_c = compute_rewards(enc_c["input_ids"], enc_c["attention_mask"])
        r_r = compute_rewards(enc_r["input_ids"], enc_r["attention_mask"])
        return r_c.detach().cpu().tolist(), r_r.detach().cpu().tolist(), trunc_c, trunc_r

    was_training_bb = backbone.training
    was_training_rh = reward_head.training
    backbone.eval()
    reward_head.eval()

    accepted_scores: List[float] = []
    rejected_scores: List[float] = []
    truncated_c: List[bool] = []
    truncated_r: List[bool] = []

    # Retry once at half batch size on OOM, mirroring the old subprocess hedge.
    bs = args.eval_batch_size
    try:
        with _torch.no_grad():
            for i in range(0, n, bs):
                a, r, tc, tr = score_batch(pairs[i:i + bs], bs)
                accepted_scores.extend(a); rejected_scores.extend(r)
                truncated_c.extend(tc); truncated_r.extend(tr)
    except (_torch.cuda.OutOfMemoryError, RuntimeError) as e:
        msg = str(e).lower()
        if "out of memory" not in msg and not isinstance(e, _torch.cuda.OutOfMemoryError):
            raise
        _try_empty_cache()
        bs = max(1, args.eval_batch_size // 2)
        print(f"[eval:{tag}] OOM at batch_size={args.eval_batch_size}; retrying at {bs}")
        accepted_scores.clear(); rejected_scores.clear()
        truncated_c.clear(); truncated_r.clear()
        with _torch.no_grad():
            for i in range(0, n, bs):
                a, r, tc, tr = score_batch(pairs[i:i + bs], bs)
                accepted_scores.extend(a); rejected_scores.extend(r)
                truncated_c.extend(tc); truncated_r.extend(tr)

    if was_training_bb:
        backbone.train()
    if was_training_rh:
        reward_head.train()

    # Aggregate metrics — same keys as evaluate_reward_model.py's summarize_pairwise_results.
    tie_eps = 1e-6
    correct = 0
    ties = 0
    margins: List[float] = []
    half_credit: List[float] = []
    for sc, sr in zip(accepted_scores, rejected_scores):
        m = sc - sr
        margins.append(m)
        if m > tie_eps:
            correct += 1
            half_credit.append(1.0)
        elif m < -tie_eps:
            half_credit.append(0.0)
        else:
            ties += 1
            half_credit.append(0.5)

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    metrics = {
        "pairwise_accuracy": (correct / n) if n else 0.0,
        "pairwise_accuracy_ties_half_credit": _mean(half_credit),
        "tie_rate": (ties / n) if n else 0.0,
        "mean_score_margin": _mean(margins),
        "mean_accepted_score": _mean(accepted_scores),
        "mean_rejected_score": _mean(rejected_scores),
        "preference_log_loss": _mean([_logistic_loss(m) for m in margins]),
        "truncated_pair_rate": (
            sum(1 for tc, tr in zip(truncated_c, truncated_r) if tc or tr) / n
        ) if n else 0.0,
    }

    output = {
        "task": "reward_model_pairwise_preference_eval",
        "metrics": metrics,
        "num_total": n,
        "num_correct": correct,
        "num_incorrect": n - correct - ties,
        "num_ties": ties,
        "dataset_alias": dataset_alias,
        "csv_path": str(csv_path),
        "num_pairs_requested": num_pairs,
        "eval_batch_size_used": bs,
        "eval_max_length": args.eval_max_length,
        "exact_duplicate_rows_skipped": skipped_duplicates,
        "eval_timestamp": now_iso(),
    }
    atomic_write_json(output_json, output)

    print(f"[eval:{tag}] pairwise_acc={metrics['pairwise_accuracy']:.3f} "
          f"margin={metrics['mean_score_margin']:+.3f} "
          f"tie_rate={metrics['tie_rate']:.3f} ({correct}/{n} correct)")

    return {
        "status": "succeeded",
        "pairwise_accuracy": metrics["pairwise_accuracy"],
        "pairwise_accuracy_ties_half_credit": metrics["pairwise_accuracy_ties_half_credit"],
        "tie_rate": metrics["tie_rate"],
        "mean_score_margin": metrics["mean_score_margin"],
        "preference_log_loss": metrics["preference_log_loss"],
        "truncated_pair_rate": metrics["truncated_pair_rate"],
        "num_pairs": n,
        "num_correct": correct,
        "num_ties": ties,
        "batch_size_used": bs,
    }


# ============================================================
# Augmenting subprocess eval via evaluate_reward_model.py
#
# Each run also invokes the submodule's evaluate_reward_model.py per
# dataset and writes eval_rm_<alias>.json next to eval_<alias>.json so
# the two metric paths can be cross-checked. The script auto-detects
# reward_head.pt inside --model_path (added in submodule commit b6eb08a).
# Failures here are logged but do NOT fail the run — script eval is
# supplementary, not the source of truth.
# ============================================================

EVAL_SCRIPT_PATH = SUBMODULE_ROOT / "evaluate_reward_model.py"


def script_eval_output_path(run_dir: Path, dataset_alias: str) -> Path:
    return run_dir / f"eval_rm_{dataset_alias}.json"


def run_script_eval(
    *,
    dataset_alias: str,
    csv_path: Path,
    ckpt_dir: Path,
    out_json: Path,
    base_model: str,
    args: argparse.Namespace,
    num_pairs: Optional[int],
    tag: str,
) -> Optional[Dict[str, Any]]:
    """Invoke evaluate_reward_model.py for one dataset; return its metrics dict."""
    if out_json.exists():
        try:
            return load_json(out_json).get("metrics") or {}
        except Exception:
            print(f"[script_eval:{tag}] existing {out_json.name} unreadable; re-running")

    if not EVAL_SCRIPT_PATH.exists():
        print(f"[script_eval:{tag}] {EVAL_SCRIPT_PATH} missing; skipping (submodule not initialized?).")
        return None

    cmd = [
        sys.executable, "-u", str(EVAL_SCRIPT_PATH),
        "--base_model", base_model,
        "--model_path", str(ckpt_dir),
        "--custom_csv", str(csv_path),
        "--output", str(out_json),
        "--max_length", str(args.eval_max_length),
        "--batch_size", str(args.eval_batch_size),
        "--torch_dtype", args.torch_dtype,
        "--device_map", "auto",
        # Pipeline trains and evaluates with NO system prompt; pass an empty
        # string explicitly so the script does not fall back to its dataset
        # default (see risk_averse_prompts.resolve_system_prompt).
        "--system_prompt", "",
    ]
    if num_pairs:
        cmd.extend(["--num_pairs", str(num_pairs)])
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")

    print(f"[script_eval:{tag}] {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, cwd=str(SUBMODULE_ROOT), check=True)
    except subprocess.CalledProcessError as e:
        print(f"[script_eval:{tag}] subprocess failed (exit={e.returncode}); pipeline continues.")
        return None
    except Exception as e:
        print(f"[script_eval:{tag}] error invoking script: {type(e).__name__}: {e}")
        return None

    try:
        return load_json(out_json).get("metrics") or {}
    except Exception as e:
        print(f"[script_eval:{tag}] could not parse output JSON {out_json}: {e}")
        return None


# ============================================================
# Status tracking
# ============================================================

@dataclass
class RunKey:
    phase: str
    lr: float
    seed: int

    def as_tuple(self) -> Tuple[str, float, int]:
        return (self.phase, round(self.lr, 12), self.seed)


def run_dir_for(output_dir: Path, key: RunKey) -> Path:
    return output_dir / key.phase / f"lr_{lr_tag(key.lr)}_seed_{key.seed}"


def init_status(args: argparse.Namespace, candidate_lrs: List[float], planned: List[RunKey]) -> Dict[str, Any]:
    return {
        "base_model": args.base_model,
        "output_dir": str(args.output_dir),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "lr_ladder": LR_LADDER,
        "middle_lr": args.middle_lr,
        "candidate_lrs": candidate_lrs,
        "best_lr": args.best_lr if args.skip_validation else None,
        "counts": {"planned": len(planned), "succeeded": 0, "failed": 0, "pending": len(planned)},
        "runs": [
            {
                "phase": k.phase, "lr": k.lr, "seed": k.seed,
                "run_dir": str(run_dir_for(args.output_dir, k).relative_to(args.output_dir)),
                "status": "pending",
                "training_wall_seconds": None,
                "eval": {},
                "error_type": None, "error_message": None, "traceback": None,
            }
            for k in planned
        ],
    }


def update_status_counts(status: Dict[str, Any]) -> None:
    c = {"planned": len(status["runs"]), "succeeded": 0, "failed": 0, "pending": 0, "running": 0}
    for r in status["runs"]:
        s = r.get("status", "pending")
        if s in c:
            c[s] += 1
    status["counts"] = c


def find_run(status: Dict[str, Any], key: RunKey) -> Dict[str, Any]:
    for r in status["runs"]:
        if r["phase"] == key.phase and abs(r["lr"] - key.lr) < 1e-12 and r["seed"] == key.seed:
            return r
    raise KeyError(f"no status entry for {key}")


def write_status(status: Dict[str, Any], path: Path) -> None:
    status["updated_at"] = now_iso()
    update_status_counts(status)
    atomic_write_json(path, status)


# ============================================================
# Phase runners
# ============================================================

def run_single(
    key: RunKey,
    base_model: str,
    train_df: pd.DataFrame,
    eval_datasets: List[str],
    num_pairs_eval: int,
    args: argparse.Namespace,
    status: Dict[str, Any],
    status_path: Path,
) -> None:
    run_dir = run_dir_for(args.output_dir, key)
    entry = find_run(status, key)
    entry["status"] = "running"
    write_status(status, status_path)

    run_start = time.time()
    model_objs: Optional[Dict[str, Any]] = None
    try:
        # --- Training (skip if complete.json exists) ---
        complete_path = run_dir / "complete.json"
        if complete_path.exists():
            print(f"[run] {key} training already complete; skipping.")
            train_info = load_json(complete_path)
        else:
            train_info, model_objs = train_one_run(base_model, key.lr, key.seed, train_df, run_dir, args)

        entry["training_wall_seconds"] = train_info.get("wall_seconds")

        # --- Eval (skip per-dataset if output JSON exists) ---
        pending_evals = [ds for ds in eval_datasets if not (run_dir / f"eval_{ds}.json").exists()]
        if pending_evals and model_objs is None:
            print(f"[run] {key} loading checkpoint for in-process eval ({len(pending_evals)} datasets pending)")
            model_objs = load_checkpoint_for_eval(base_model, run_dir / "checkpoint", args)

        for ds in eval_datasets:
            out_json = run_dir / f"eval_{ds}.json"
            entry_eval = entry["eval"].setdefault(ds, {})
            if out_json.exists():
                print(f"[run] {key} {ds} already evaluated; loading existing.")
                data = load_json(out_json)
                metrics = data.get("metrics", {})
                entry_eval.update({
                    "status": "succeeded",
                    "pairwise_accuracy": metrics.get("pairwise_accuracy"),
                    "pairwise_accuracy_ties_half_credit": metrics.get("pairwise_accuracy_ties_half_credit"),
                    "tie_rate": metrics.get("tie_rate"),
                    "mean_score_margin": metrics.get("mean_score_margin"),
                    "preference_log_loss": metrics.get("preference_log_loss"),
                    "truncated_pair_rate": metrics.get("truncated_pair_rate"),
                    "num_pairs": data.get("num_total"),
                    "num_correct": data.get("num_correct"),
                    "num_ties": data.get("num_ties", 0),
                })
            else:
                tag = f"{key.phase}_lr_{lr_tag(key.lr)}_seed_{key.seed}_{ds}"
                result = score_pairs_in_process(
                    model_objs=model_objs,
                    dataset_alias=ds,
                    num_pairs=num_pairs_eval,
                    output_json=out_json,
                    args=args,
                    tag=tag,
                )
                entry_eval.update(result)

            # Augment with subprocess script eval (cross-checks the in-process
            # numbers above against evaluate_reward_model.py's loader path).
            if not args.skip_script_eval:
                rm_out = script_eval_output_path(run_dir, ds)
                rel = DATASET_ALIAS_PATHS.get(ds)
                if rel is None:
                    print(f"[script_eval] no CSV path for dataset {ds!r}; skipping.")
                else:
                    rm_metrics = run_script_eval(
                        dataset_alias=ds,
                        csv_path=SUBMODULE_ROOT / rel,
                        ckpt_dir=run_dir / "checkpoint",
                        out_json=rm_out,
                        base_model=base_model,
                        args=args,
                        num_pairs=num_pairs_eval,
                        tag=f"{key.phase}_lr_{lr_tag(key.lr)}_seed_{key.seed}_{ds}",
                    )
                    if rm_metrics:
                        entry_eval["script_eval"] = {
                            "pairwise_accuracy": rm_metrics.get("pairwise_accuracy"),
                            "pairwise_accuracy_ties_half_credit":
                                rm_metrics.get("pairwise_accuracy_ties_half_credit"),
                            "mean_score_margin": rm_metrics.get("mean_score_margin"),
                            "tie_rate": rm_metrics.get("tie_rate"),
                            "preference_log_loss": rm_metrics.get("preference_log_loss"),
                            "truncated_pair_rate": rm_metrics.get("truncated_pair_rate"),
                            "output_json": str(rm_out),
                        }
            write_status(status, status_path)

        entry["status"] = "succeeded"
        entry["error_type"] = None
        entry["error_message"] = None
        entry["traceback"] = None

    except OOMSkipRun as e:
        entry["status"] = "failed"
        entry["error_type"] = "OOMSkipRun"
        entry["error_message"] = str(e)
        entry["traceback"] = traceback.format_exc()
        print(f"[run] {key} failed (OOM): {e}")
    except Exception as e:
        entry["status"] = "failed"
        entry["error_type"] = type(e).__name__
        entry["error_message"] = str(e)
        entry["traceback"] = traceback.format_exc()
        print(f"[run] {key} failed: {type(e).__name__}: {e}")
    finally:
        entry["wall_seconds"] = round(time.time() - run_start, 1)
        # Explicit cleanup so the next run starts with a clean GPU.
        if model_objs is not None:
            for k in ("backbone", "reward_head", "tokenizer", "compute_rewards"):
                model_objs.pop(k, None)
            del model_objs
        gc.collect()
        _try_empty_cache()
        write_status(status, status_path)


def choose_best_lr(status: Dict[str, Any], candidate_lrs: List[float]) -> Optional[float]:
    """Pick best LR from validation phase by pairwise_accuracy_ties_half_credit,
    tiebreak by higher mean_score_margin."""
    scored: List[Tuple[float, float, float]] = []
    for lr in candidate_lrs:
        key = RunKey("validation", lr, 1)
        try:
            entry = find_run(status, key)
        except KeyError:
            continue
        if entry["status"] != "succeeded":
            continue
        val_eval = entry["eval"].get("reward_model_validation", {})
        acc = val_eval.get("pairwise_accuracy_ties_half_credit")
        margin = val_eval.get("mean_score_margin")
        if acc is None:
            continue
        scored.append((lr, float(acc), float(margin) if margin is not None else 0.0))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
    return scored[0][0]


# ============================================================
# Main
# ============================================================

def build_planned_runs(args: argparse.Namespace, candidate_lrs: List[float]) -> List[RunKey]:
    planned: List[RunKey] = []
    if not args.skip_validation:
        for lr in candidate_lrs:
            for seed in args.seeds_validation:
                planned.append(RunKey("validation", lr, seed))
    if not args.skip_heldout:
        best_lr = args.best_lr if args.skip_validation else candidate_lrs[len(candidate_lrs) // 2]
        for seed in args.seeds_heldout:
            planned.append(RunKey("heldout", best_lr, seed))
    return planned


def parse_int_csv(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_str_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    p.add_argument("--middle_lr", type=float, default=DEFAULT_MIDDLE_LR,
                   help=f"Middle LR rung (default {DEFAULT_MIDDLE_LR:.0e}); must be in {LR_LADDER}")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--train_csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--val_num_pairs", type=int, default=DEFAULT_VAL_NUM_PAIRS)
    p.add_argument("--heldout_num_pairs", type=int, default=DEFAULT_HELDOUT_NUM_PAIRS)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad_accum_steps", type=int, default=DEFAULT_GRAD_ACCUM_STEPS)
    p.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--max_grad_norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    p.add_argument("--eval_batch_size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    p.add_argument("--eval_max_length", type=int, default=DEFAULT_EVAL_MAX_LENGTH)
    p.add_argument("--seeds_validation", type=parse_int_csv, default=DEFAULT_SEEDS_VALIDATION)
    p.add_argument("--seeds_heldout", type=parse_int_csv, default=DEFAULT_SEEDS_HELDOUT)
    p.add_argument("--skip_validation", action="store_true")
    p.add_argument("--single_lr", action="store_true",
                   help="Train only at --middle_lr; skip ladder neighbors. "
                        "Use after LR ablation is done, for epoch ablation at a locked LR.")
    p.add_argument("--best_lr", type=float, default=None)
    p.add_argument("--skip_heldout", action="store_true")
    p.add_argument("--heldout_datasets", type=parse_str_csv, default=DEFAULT_HELDOUT_DATASETS)
    p.add_argument(
        "--skip_script_eval",
        action="store_true",
        help=(
            "Disable the per-dataset evaluate_reward_model.py subprocess eval "
            "(otherwise each run writes eval_rm_<alias>.json next to the "
            "in-process eval_<alias>.json for cross-checking)."
        ),
    )
    p.add_argument("--torch_dtype", type=str, default=DEFAULT_TORCH_DTYPE)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--grad_ckpt", dest="grad_ckpt", action="store_true", default=DEFAULT_GRAD_CKPT)
    p.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if snap_to_ladder(args.middle_lr) is None:
        p.error(f"--middle_lr={args.middle_lr} not in ladder {LR_LADDER} (Ben.pdf 628-636).")
    if args.skip_validation:
        if args.best_lr is None:
            p.error("--skip_validation requires --best_lr")
        if snap_to_ladder(args.best_lr) is None:
            p.error(f"--best_lr={args.best_lr} not in ladder {LR_LADDER} (Ben.pdf 628-636).")

    if args.output_dir is None:
        tag = args.base_model.replace("/", "_").replace(":", "_")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.output_dir = PROJECT_ROOT / "outputs" / f"rft_{tag}_{stamp}"

    args.output_dir = Path(args.output_dir).resolve()
    args.train_csv = Path(args.train_csv).resolve()
    return args


def write_final_summary(args: argparse.Namespace, status: Dict[str, Any]) -> None:
    best_lr = status.get("best_lr")
    summary: Dict[str, Any] = {
        "base_model": args.base_model,
        "best_lr": best_lr,
        "validation": None,
        "heldout_per_seed": {},
        "heldout_aggregate": {},
    }

    if best_lr is not None:
        try:
            val_entry = find_run(status, RunKey("validation", best_lr, 1))
            val_eval = val_entry["eval"].get("reward_model_validation", {})
            summary["validation"] = {
                "lr": best_lr,
                "seed": 1,
                "pairwise_accuracy": val_eval.get("pairwise_accuracy"),
                "pairwise_accuracy_ties_half_credit": val_eval.get("pairwise_accuracy_ties_half_credit"),
                "mean_score_margin": val_eval.get("mean_score_margin"),
                "tie_rate": val_eval.get("tie_rate"),
                "num_pairs": val_eval.get("num_pairs"),
            }
        except KeyError:
            pass

        per_seed: Dict[str, Dict[str, Any]] = {}
        for seed in args.seeds_heldout:
            try:
                entry = find_run(status, RunKey("heldout", best_lr, seed))
            except KeyError:
                continue
            if entry["status"] != "succeeded":
                continue
            per_seed[str(seed)] = {ds: entry["eval"].get(ds, {}) for ds in args.heldout_datasets}
        summary["heldout_per_seed"] = per_seed

        agg: Dict[str, Dict[str, Any]] = {}
        for ds in args.heldout_datasets:
            by_metric: Dict[str, List[float]] = {
                "pairwise_accuracy": [],
                "pairwise_accuracy_ties_half_credit": [],
                "mean_score_margin": [],
                "tie_rate": [],
            }
            for seed, ds_map in per_seed.items():
                e = ds_map.get(ds, {})
                for m in by_metric:
                    v = e.get(m)
                    if v is not None:
                        by_metric[m].append(float(v))
            ds_summary: Dict[str, Any] = {"n_seeds": len(next(iter(by_metric.values()), []))}
            for m, vals in by_metric.items():
                if not vals:
                    ds_summary[m] = {"mean": None, "sd": None}
                elif len(vals) == 1:
                    ds_summary[m] = {"mean": vals[0], "sd": None}
                else:
                    ds_summary[m] = {
                        "mean": float(np.mean(vals)),
                        "sd": float(np.std(vals, ddof=1)),
                    }
            agg[ds] = ds_summary
        summary["heldout_aggregate"] = agg

    atomic_write_json(args.output_dir / "final_summary.json", summary)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidate_lrs = select_candidate_lrs(args.middle_lr)
    if args.single_lr:
        candidate_lrs = [args.middle_lr]
        print(f"[main] --single_lr set; training only at middle_lr={args.middle_lr:.1e} (no ladder neighbors).")
    if args.skip_validation:
        print(f"[main] --skip_validation set; using provided --best_lr={args.best_lr}")
    elif not args.single_lr and len(candidate_lrs) < 3:
        print(
            f"[main] WARNING: middle_lr={args.middle_lr} is at the ladder edge; only {len(candidate_lrs)} "
            f"rungs will be tested ({candidate_lrs}). Consider messaging Elliott (Ben.pdf line 656)."
        )

    atomic_write_json(
        args.output_dir / "config.json",
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    )

    planned = build_planned_runs(args, candidate_lrs)
    status_path = args.output_dir / "status.json"
    status = init_status(args, candidate_lrs, planned)

    # Reconcile with existing status.json on resume.
    if status_path.exists():
        try:
            prior = load_json(status_path)
            prior_by_key = {(r["phase"], round(r["lr"], 12), r["seed"]): r for r in prior.get("runs", [])}
            for r in status["runs"]:
                k = (r["phase"], round(r["lr"], 12), r["seed"])
                if k in prior_by_key:
                    r.update({
                        "status": prior_by_key[k].get("status", "pending"),
                        "training_wall_seconds": prior_by_key[k].get("training_wall_seconds"),
                        "eval": prior_by_key[k].get("eval", {}),
                        "error_type": prior_by_key[k].get("error_type"),
                        "error_message": prior_by_key[k].get("error_message"),
                        "traceback": prior_by_key[k].get("traceback"),
                    })
            status["best_lr"] = prior.get("best_lr", status["best_lr"])
        except Exception as e:
            print(f"[main] WARNING: could not read existing status.json ({e}); starting fresh.")

    write_status(status, status_path)

    if args.dry_run:
        print("[main] --dry_run: planned runs:")
        for k in planned:
            print(f"  {k.phase:10s}  lr={k.lr:.1e}  seed={k.seed}  -> {run_dir_for(args.output_dir, k)}")
        return 0

    print(f"[main] loading training CSV: {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    print(f"[main] training rows: {len(train_df)}")

    # --- Phase 1: Validation ---
    if not args.skip_validation:
        for lr in candidate_lrs:
            for seed in args.seeds_validation:
                run_single(
                    key=RunKey("validation", lr, seed),
                    base_model=args.base_model,
                    train_df=train_df,
                    eval_datasets=["reward_model_validation"],
                    num_pairs_eval=args.val_num_pairs,
                    args=args,
                    status=status,
                    status_path=status_path,
                )

        best_lr = choose_best_lr(status, candidate_lrs)
        if best_lr is None:
            print("[main] ERROR: no successful validation run; cannot pick best_lr.")
            write_final_summary(args, status)
            return 1
        status["best_lr"] = best_lr
        write_status(status, status_path)
        print(f"[main] best_lr = {best_lr:.1e}")
    else:
        best_lr = args.best_lr
        status["best_lr"] = best_lr
        write_status(status, status_path)

    # --- Phase 2: Held-out ---
    if not args.skip_heldout:
        # Re-plan held-out entries if best_lr changed from the middle-of-ladder placeholder.
        needed = [RunKey("heldout", best_lr, s) for s in args.seeds_heldout]
        existing_keys = {(r["phase"], round(r["lr"], 12), r["seed"]) for r in status["runs"]}
        for k in needed:
            tk = (k.phase, round(k.lr, 12), k.seed)
            if tk not in existing_keys:
                status["runs"].append({
                    "phase": k.phase, "lr": k.lr, "seed": k.seed,
                    "run_dir": str(run_dir_for(args.output_dir, k).relative_to(args.output_dir)),
                    "status": "pending",
                    "training_wall_seconds": None,
                    "eval": {},
                    "error_type": None, "error_message": None, "traceback": None,
                })
        write_status(status, status_path)

        for seed in args.seeds_heldout:
            run_single(
                key=RunKey("heldout", best_lr, seed),
                base_model=args.base_model,
                train_df=train_df,
                eval_datasets=args.heldout_datasets,
                num_pairs_eval=args.heldout_num_pairs,
                args=args,
                status=status,
                status_path=status_path,
            )

    write_final_summary(args, status)
    print(f"[main] done. Output dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
