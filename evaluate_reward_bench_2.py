#!/usr/bin/env python3
"""Evaluate a trained risk-averse reward model on allenai/reward-bench-2.

Standalone: independent of the submodule's evaluate_reward_model.py. Loads the
same checkpoint format produced by rft_pipeline.py (AutoModel + LoRA adapter
+ reward_head.pt) and writes its own JSON.

Reports two metric families in one run:

(A) Standard 5 subsets — Factuality, Precise IF, Math, Safety, Focus
    Each row has 1 chosen + 3 rejected. Reported per-subset and overall:
      all_pairs_win_accuracy  — canonical RB2 score (chosen > every rejected)
      pairwise_accuracy       — mean over 3 chosen-vs-rejected pairs per row
      mean_margin             — mean(score(chosen) - score(rejected_i))

(B) Ties subset (canonical RB2 weighted score, mirrors
    allenai/reward-bench rewardbench/utils.py::process_single_model):
    Each row has multiple chosen + multiple rejected. Rows are tagged
    "ref:<prompt_id>" or "tied:<prompt_id>" via the id field. Per-prompt:
      accurate                  worst correct > best incorrect
      diff_correct_margin       best correct - worst correct (None if 1 correct)
      correct_incorrect_margin  worst correct - best incorrect
    Aggregates:
      ref_accuracy              mean accurate over ref rows
      tied_accuracy             mean accurate over tied rows
      correctness_preferred       mean(corr_inc_tied  > diff_corr) over shared ids
      correctness_preferred_hard  mean(min(corr_inc_ref, corr_inc_tied) > diff_corr)
      correctness_margin_score    tanh tiebreaker
      overall_score = 0.30*tied_acc + 0.30*ref_acc + 0.20*corr_pref
                    + 0.20*corr_pref_hard + 0.01*margin_score

Combined RB2 leaderboard score = mean of the 5 standard all_pairs_win
accuracies + the Ties overall_score (6 numbers averaged).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STANDARD_SUBSETS = ["Factuality", "Precise IF", "Math", "Safety", "Focus"]
TIES_SUBSET_NAME = "Ties"
ALL_SUBSETS_DEFAULT = STANDARD_SUBSETS + [TIES_SUBSET_NAME]


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def parse_torch_dtype(name: str):
    import torch
    n = str(name).strip().lower()
    if n == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return {
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
    }[n]


def unwrap_text_backbone(backbone):
    """Mirror rft_pipeline.unwrap_text_backbone: strip vision tower for Gemma 3 etc.

    The trained LoRA adapter targets the *unwrapped* text submodule, so the
    PEFT load below MUST be applied to the same submodule that training used.
    """
    for attr in ("language_model", "text_model"):
        sub = getattr(backbone, attr, None)
        if sub is not None and hasattr(sub.config, "hidden_size"):
            print(f"[rb2] unwrap_text_backbone: using .{attr} ({type(sub).__name__})", flush=True)
            text_model = sub
            del backbone
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return text_model
    return backbone


def load_reward_model(base_model: str, model_path: str, *, reward_head_path: Optional[str],
                      torch_dtype: str, device_map: str, trust_remote_code: bool):
    """Return (compute_rewards_fn, tokenizer). Mirrors rft_pipeline forward path.

    compute_rewards_fn(input_ids, attention_mask) -> 1-D float tensor of rewards.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=trust_remote_code, token=_hf_token()
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    tok.padding_side = "right"

    dtype = parse_torch_dtype(torch_dtype)
    backbone = AutoModel.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        token=_hf_token(),
    )
    backbone = unwrap_text_backbone(backbone)
    if getattr(backbone.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        backbone.config.pad_token_id = tok.pad_token_id

    backbone = PeftModel.from_pretrained(backbone, model_path)
    backbone.eval()

    rh_path = reward_head_path or os.path.join(model_path, "reward_head.pt")
    if not os.path.exists(rh_path):
        raise FileNotFoundError(f"reward_head.pt not found at {rh_path}")
    ckpt = torch.load(rh_path, map_location="cpu")
    state = ckpt["reward_head_state_dict"]
    out_features, in_features = state["weight"].shape
    has_bias = "bias" in state
    head = torch.nn.Linear(in_features, out_features, bias=has_bias).to(torch.float32)
    head.load_state_dict(state)

    hidden_size = getattr(backbone.config, "hidden_size", None)
    if hidden_size is not None and in_features != hidden_size:
        raise ValueError(
            f"reward_head hidden_size={in_features} mismatches backbone hidden_size={hidden_size}"
        )
    device = next(backbone.parameters()).device
    head = head.to(device)

    @torch.no_grad()
    def compute_rewards(input_ids, attention_mask):
        out = backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = out.last_hidden_state  # (B, T, H)
        seq_lens = attention_mask.sum(dim=1) - 1
        B = input_ids.shape[0]
        last = hidden[torch.arange(B, device=hidden.device), seq_lens].float()
        return head(last).squeeze(-1)

    return compute_rewards, tok, device


def format_chat(tokenizer, prompt: str, response: str) -> str:
    """Match training: [user, assistant] turns, no system prompt."""
    messages = [
        {"role": "user", "content": str(prompt)},
        {"role": "assistant", "content": str(response)},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False)


def load_rb2(cache_dir: Optional[str], subsets: List[str], num_examples: Optional[int]
             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (standard_rows, ties_rows).

    standard_rows: {id, prompt, chosen, rejecteds, subset} — 1 chosen + N rejected
    ties_rows:     {id, prompt, candidates, num_correct, subset, sample_type, prompt_id}
                   candidates = chosen ++ rejected; first num_correct are correct.

    Filters to `subsets` (Ties handled separately from the other 5) and
    (optionally) caps total combined count to `num_examples`.
    """
    from datasets import load_dataset
    print(f"[rb2] loading allenai/reward-bench-2 (cache_dir={cache_dir})", flush=True)
    ds = load_dataset("allenai/reward-bench-2", cache_dir=cache_dir)
    if hasattr(ds, "keys"):
        split_names = list(ds.keys())
        if len(split_names) == 1:
            ds = ds[split_names[0]]
        elif "test" in split_names:
            ds = ds["test"]
        else:
            ds = ds[split_names[0]]

    standard_rows: List[Dict[str, Any]] = []
    ties_rows: List[Dict[str, Any]] = []
    subsets_set = set(subsets)
    seen = 0
    for ex in ds:
        if ex["subset"] not in subsets_set:
            continue
        chosen_list = list(ex.get("chosen") or [])
        rejected_list = list(ex.get("rejected") or [])
        if not chosen_list or not rejected_list:
            continue

        if ex["subset"] == TIES_SUBSET_NAME:
            # id is "<sample_type>:<prompt_id>"; sample_type ∈ {"ref","tied"}
            raw_id = str(ex.get("id", ""))
            if ":" not in raw_id:
                # malformed; can't compute ref/tied groupings — skip
                continue
            sample_type, prompt_id_str = raw_id.split(":", 1)
            try:
                prompt_id = int(prompt_id_str)
            except ValueError:
                prompt_id = prompt_id_str  # fall back to string
            num_correct = int(ex.get("num_correct", len(chosen_list)))
            ties_rows.append({
                "id": raw_id,
                "prompt": ex["prompt"],
                "candidates": chosen_list + rejected_list,  # first num_correct are correct
                "num_correct": num_correct,
                "subset": TIES_SUBSET_NAME,
                "sample_type": sample_type,
                "prompt_id": prompt_id,
            })
        else:
            standard_rows.append({
                "id": ex.get("id"),
                "prompt": ex["prompt"],
                "chosen": chosen_list[0],          # canonical RB2: 1 chosen
                "rejecteds": rejected_list,        # canonical RB2: 3 rejecteds
                "subset": ex["subset"],
            })

        seen += 1
        if num_examples is not None and seen >= num_examples:
            break

    print(f"[rb2] loaded {len(standard_rows)} standard rows + {len(ties_rows)} ties rows "
          f"across subsets {sorted(subsets_set)}", flush=True)
    return standard_rows, ties_rows


def build_inputs(rows: List[Dict[str, Any]], tokenizer, max_length: int) -> List[Dict[str, Any]]:
    """Flatten standard rows to (row_idx, kind, j, input_ids, length) entries.

    kind is "chosen" or "rejected"; j is the rejected index (0 for chosen).
    Sorted by length to reduce padding waste during batched scoring.
    """
    items: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        c_text = format_chat(tokenizer, row["prompt"], row["chosen"])
        c_enc = tokenizer(c_text, truncation=True, max_length=max_length, add_special_tokens=False)
        items.append({"row_idx": i, "kind": "chosen", "j": 0,
                      "input_ids": c_enc["input_ids"], "length": len(c_enc["input_ids"]),
                      "truncated": len(c_enc["input_ids"]) >= max_length})
        for j, r in enumerate(row["rejecteds"]):
            r_text = format_chat(tokenizer, row["prompt"], r)
            r_enc = tokenizer(r_text, truncation=True, max_length=max_length, add_special_tokens=False)
            items.append({"row_idx": i, "kind": "rejected", "j": j,
                          "input_ids": r_enc["input_ids"], "length": len(r_enc["input_ids"]),
                          "truncated": len(r_enc["input_ids"]) >= max_length})
    items.sort(key=lambda x: x["length"])
    return items


def build_inputs_ties(rows: List[Dict[str, Any]], tokenizer, max_length: int) -> List[Dict[str, Any]]:
    """Flatten ties rows to one item per candidate. Each item knows its
    row_idx, candidate_idx, and is_correct (idx < num_correct).
    """
    items: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        for k, cand in enumerate(row["candidates"]):
            text = format_chat(tokenizer, row["prompt"], cand)
            enc = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=False)
            items.append({
                "row_idx": i,
                "candidate_idx": k,
                "is_correct": k < row["num_correct"],
                "input_ids": enc["input_ids"],
                "length": len(enc["input_ids"]),
                "truncated": len(enc["input_ids"]) >= max_length,
            })
    items.sort(key=lambda x: x["length"])
    return items


def score_items(items: List[Dict[str, Any]], compute_rewards, tokenizer, device, batch_size: int) -> List[float]:
    import torch
    pad_id = tokenizer.pad_token_id
    scores: List[Optional[float]] = [None] * len(items)
    n_batches = math.ceil(len(items) / batch_size)
    t0 = time.time()
    for b in range(n_batches):
        chunk = items[b * batch_size: (b + 1) * batch_size]
        max_len = max(it["length"] for it in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for i, it in enumerate(chunk):
            ids = it["input_ids"]
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        rewards = compute_rewards(input_ids, attention_mask).detach().cpu().tolist()
        for it, r in zip(chunk, rewards):
            # items are original positions in the flat list; preserve via embedded index
            scores[it["_orig_idx"]] = float(r)
        if (b + 1) % 20 == 0 or b == n_batches - 1:
            done = (b + 1) * batch_size
            elapsed = time.time() - t0
            print(f"[rb2] scored {min(done, len(items))}/{len(items)} (batch {b+1}/{n_batches}, "
                  f"{elapsed:.1f}s elapsed)", flush=True)
    return [s for s in scores if s is not None]


def aggregate(rows: List[Dict[str, Any]], items: List[Dict[str, Any]],
              tie_epsilon: float = 1e-6) -> Dict[str, Any]:
    """Group flat scores by row, compute per-subset and overall metrics."""
    by_row: Dict[int, Dict[str, Any]] = {i: {"chosen": None, "rejecteds": [], "truncated": False}
                                          for i in range(len(rows))}
    for it in items:
        s = it["_score"]
        by_row[it["row_idx"]]["truncated"] = by_row[it["row_idx"]]["truncated"] or it["truncated"]
        if it["kind"] == "chosen":
            by_row[it["row_idx"]]["chosen"] = s
        else:
            # ensure stable ordering by j
            slot = by_row[it["row_idx"]]
            while len(slot["rejecteds"]) <= it["j"]:
                slot["rejecteds"].append(None)
            slot["rejecteds"][it["j"]] = s

    per_subset: Dict[str, Dict[str, Any]] = {}
    overall_pair_correct = 0
    overall_pair_total = 0
    overall_row_correct = 0
    overall_row_total = 0
    overall_margins: List[float] = []
    overall_truncated = 0
    per_example: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        agg_row = by_row[i]
        c = agg_row["chosen"]
        rs = [r for r in agg_row["rejecteds"] if r is not None]
        if c is None or not rs:
            continue
        margins = [c - r for r in rs]
        wins = [m > tie_epsilon for m in margins]
        all_win = all(wins)
        pair_correct = sum(1 for w in wins if w)
        pair_total = len(wins)

        sub = row["subset"]
        sd = per_subset.setdefault(sub, {
            "n_examples": 0, "n_examples_all_pairs_win": 0,
            "n_pairs": 0, "n_pairs_correct": 0,
            "margins": [], "truncated_examples": 0,
        })
        sd["n_examples"] += 1
        sd["n_examples_all_pairs_win"] += 1 if all_win else 0
        sd["n_pairs"] += pair_total
        sd["n_pairs_correct"] += pair_correct
        sd["margins"].extend(margins)
        if agg_row["truncated"]:
            sd["truncated_examples"] += 1
            overall_truncated += 1

        overall_pair_correct += pair_correct
        overall_pair_total += pair_total
        overall_row_correct += 1 if all_win else 0
        overall_row_total += 1
        overall_margins.extend(margins)

        per_example.append({
            "id": row["id"],
            "subset": sub,
            "chosen_score": c,
            "rejected_scores": rs,
            "margins": margins,
            "all_pairs_win": all_win,
            "truncated": agg_row["truncated"],
        })

    metrics_per_subset: Dict[str, Dict[str, Any]] = {}
    for sub, sd in per_subset.items():
        margins = sd["margins"]
        metrics_per_subset[sub] = {
            "all_pairs_win_accuracy": sd["n_examples_all_pairs_win"] / sd["n_examples"]
                if sd["n_examples"] else None,
            "pairwise_accuracy": sd["n_pairs_correct"] / sd["n_pairs"] if sd["n_pairs"] else None,
            "mean_margin": sum(margins) / len(margins) if margins else None,
            "n_examples": sd["n_examples"],
            "n_pairs": sd["n_pairs"],
            "truncated_example_rate": sd["truncated_examples"] / sd["n_examples"]
                if sd["n_examples"] else None,
        }

    metrics_overall = {
        "all_pairs_win_accuracy": overall_row_correct / overall_row_total if overall_row_total else None,
        "pairwise_accuracy": overall_pair_correct / overall_pair_total if overall_pair_total else None,
        "mean_margin": sum(overall_margins) / len(overall_margins) if overall_margins else None,
        "n_examples": overall_row_total,
        "n_pairs": overall_pair_total,
        "truncated_example_rate": overall_truncated / overall_row_total if overall_row_total else None,
    }

    return {
        "metrics_overall": metrics_overall,
        "metrics_per_subset": metrics_per_subset,
        "per_example": per_example,
    }


def _compute_ties_prompt_stats(samples: List[Tuple[bool, float]]) -> Tuple[bool, Optional[float], float]:
    """Mirror rewardbench/utils.py::_compute_prompt_stats.

    Returns (accurate, diff_correct_margin, correct_incorrect_margin).
      accurate                  worst correct > best incorrect
      diff_correct_margin       best correct - worst correct (None if 1 correct)
      correct_incorrect_margin  worst correct - best incorrect
    """
    correct_scores = [s for is_corr, s in samples if is_corr]
    incorrect_scores = [s for is_corr, s in samples if not is_corr]
    best_correct = max(correct_scores)
    worst_correct = min(correct_scores)
    best_incorrect = max(incorrect_scores)
    diff_correct_margin = (best_correct - worst_correct) if len(correct_scores) > 1 else None
    correct_incorrect_margin = worst_correct - best_incorrect
    accurate = correct_incorrect_margin > 0
    return accurate, diff_correct_margin, correct_incorrect_margin


def aggregate_ties(rows: List[Dict[str, Any]], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the canonical RB2 Ties metric.

    Mirrors allenai/reward-bench rewardbench/utils.py::process_single_model.
    """
    # Group items back to (row_idx, [(is_correct, score)])
    by_row: Dict[int, List[Tuple[bool, float]]] = defaultdict(list)
    truncated_by_row: Dict[int, bool] = defaultdict(bool)
    for it in items:
        by_row[it["row_idx"]].append((bool(it["is_correct"]), float(it["_score"])))
        truncated_by_row[it["row_idx"]] = truncated_by_row[it["row_idx"]] or bool(it["truncated"])

    # Per-row stats, separated into ref / tied
    ref_stats: Dict[Any, Tuple[bool, Optional[float], float]] = {}
    tied_stats: Dict[Any, Tuple[bool, Optional[float], float]] = {}
    per_example: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        samples = by_row.get(i)
        if not samples:
            continue
        # Need at least 1 correct + 1 incorrect to compute correct_incorrect_margin
        if not any(c for c, _ in samples) or not any(not c for c, _ in samples):
            continue
        accurate, diff_corr, corr_inc = _compute_ties_prompt_stats(samples)
        bucket = ref_stats if row["sample_type"] == "ref" else tied_stats
        bucket[row["prompt_id"]] = (accurate, diff_corr, corr_inc)
        per_example.append({
            "id": row["id"],
            "sample_type": row["sample_type"],
            "prompt_id": row["prompt_id"],
            "num_correct": row["num_correct"],
            "num_total": len(row["candidates"]),
            "accurate": accurate,
            "diff_correct_margin": diff_corr,
            "correct_incorrect_margin": corr_inc,
            "truncated": truncated_by_row[i],
        })

    n_ref = len(ref_stats)
    n_tied = len(tied_stats)
    ref_accuracy = (sum(s[0] for s in ref_stats.values()) / n_ref) if n_ref else 0.0
    tied_accuracy = (sum(s[0] for s in tied_stats.values()) / n_tied) if n_tied else 0.0

    # correctness_preferred metrics defined only over prompts present in BOTH ref and tied
    shared_prompts = sorted(set(ref_stats) & set(tied_stats))

    if shared_prompts:
        diff_corr_tied = [tied_stats[pid][1] for pid in shared_prompts]
        corr_inc_tied = [tied_stats[pid][2] for pid in shared_prompts]
        corr_inc_ref = [ref_stats[pid][2] for pid in shared_prompts]
        # diff_corr_tied may be None for tied rows with only 1 correct — drop those
        usable = [k for k, v in enumerate(diff_corr_tied) if v is not None]
        if usable:
            dt = [diff_corr_tied[k] for k in usable]
            cit = [corr_inc_tied[k] for k in usable]
            cir = [corr_inc_ref[k] for k in usable]

            correctness_preferred = sum(1 for c, d in zip(cit, dt) if c > d) / len(usable)
            min_pair = [min(a, b) for a, b in zip(cir, cit)]
            correctness_preferred_hard = sum(1 for m, d in zip(min_pair, dt) if m > d) / len(usable)

            # tanh tiebreaker; nan-safe (divide-by-zero in original implementation
            # was handled via np.nan_to_num)
            margin_terms: List[float] = []
            for m, d in zip(min_pair, dt):
                if d == 0:
                    margin_terms.append(0.0)
                else:
                    margin_terms.append(math.tanh(m / d - 1.0))
            correctness_margin_score = sum(margin_terms) / len(margin_terms)
            n_shared_usable = len(usable)
        else:
            correctness_preferred = 0.0
            correctness_preferred_hard = 0.0
            correctness_margin_score = 0.0
            n_shared_usable = 0
    else:
        correctness_preferred = 0.0
        correctness_preferred_hard = 0.0
        correctness_margin_score = 0.0
        n_shared_usable = 0

    overall_score = (
        0.30 * tied_accuracy
        + 0.30 * ref_accuracy
        + 0.20 * correctness_preferred
        + 0.20 * correctness_preferred_hard
        + 0.01 * correctness_margin_score
    )

    truncated_count = sum(1 for v in truncated_by_row.values() if v)
    return {
        "ties_metrics": {
            "n_examples": n_ref + n_tied,
            "n_ref_rows": n_ref,
            "n_tied_rows": n_tied,
            "n_shared_prompts": len(shared_prompts),
            "n_shared_prompts_usable": n_shared_usable,
            "ref_accuracy": ref_accuracy,
            "tied_accuracy": tied_accuracy,
            "correctness_preferred": correctness_preferred,
            "correctness_preferred_hard": correctness_preferred_hard,
            "correctness_margin_score": correctness_margin_score,
            "overall_score": overall_score,
            "truncated_example_rate": (truncated_count / (n_ref + n_tied)) if (n_ref + n_tied) else None,
        },
        "ties_per_example": per_example,
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def existing_output_is_complete(out_path: Path, want_ties: bool) -> bool:
    """Return True if `out_path` already contains everything we'd compute now.

    Used to make re-runs idempotent: a v1 JSON (standard 5 only, missing
    `ties_metrics`) is treated as INCOMPLETE so a re-run will fill it in,
    while a v2 JSON (with `ties_metrics` present) is reused as-is.
    """
    if not out_path.exists():
        return False
    try:
        existing = json.loads(out_path.read_text())
    except Exception:
        return False
    has_standard = bool(existing.get("metrics_per_subset"))
    if not has_standard:
        return False
    if want_ties:
        return bool(existing.get("ties_metrics"))
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", required=True, help="HF base model id (e.g. google/gemma-3-12b-it)")
    p.add_argument("--model_path", required=True, help="LoRA adapter dir containing reward_head.pt")
    p.add_argument("--reward_head_path", default=None, help="Override reward_head.pt path")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--torch_dtype", default="auto")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--num_examples", type=int, default=None,
                   help="Cap total examples (debug). Default: all")
    p.add_argument("--subsets", default=",".join(ALL_SUBSETS_DEFAULT),
                   help=f"Comma-separated subset names. Default: all 6 ({ALL_SUBSETS_DEFAULT}).")
    p.add_argument("--skip_ties", action="store_true",
                   help=f"Skip the {TIES_SUBSET_NAME} subset entirely (don't load, don't score).")
    p.add_argument("--include_ties", action="store_true",
                   help="(Deprecated; Ties is included by default with proper canonical scoring.)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output JSON even if it looks complete.")
    p.add_argument("--cache_dir", default=None,
                   help="HF datasets cache_dir (default: HF default ~/.cache/huggingface/datasets)")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--tie_epsilon", type=float, default=1e-6)
    args = p.parse_args()

    requested_subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    if args.skip_ties and TIES_SUBSET_NAME in requested_subsets:
        requested_subsets = [s for s in requested_subsets if s != TIES_SUBSET_NAME]
    want_ties = TIES_SUBSET_NAME in requested_subsets
    if args.include_ties:
        print("[rb2] note: --include_ties is now the default (Ties is scored canonically)", flush=True)

    out_path = Path(args.output)
    if not args.force and existing_output_is_complete(out_path, want_ties=want_ties):
        print(f"[rb2] output {out_path} already complete (has metrics_per_subset"
              f"{' + ties_metrics' if want_ties else ''}); skipping (--force to re-run)", flush=True)
        return 0
    if out_path.exists() and not args.force:
        print(f"[rb2] output {out_path} exists but is incomplete; will re-compute and overwrite", flush=True)

    standard_rows, ties_rows = load_rb2(args.cache_dir, requested_subsets, args.num_examples)
    if not standard_rows and not ties_rows:
        print(f"[rb2] no rows loaded (subsets={requested_subsets}); aborting", flush=True)
        return 1

    print(f"[rb2] loading reward model: base={args.base_model} adapter={args.model_path}", flush=True)
    compute_rewards, tokenizer, device = load_reward_model(
        base_model=args.base_model,
        model_path=args.model_path,
        reward_head_path=args.reward_head_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    # ---- Standard 5 subsets ----
    standard_results: Dict[str, Any] = {}
    if standard_rows:
        items = build_inputs(standard_rows, tokenizer, args.max_length)
        for idx, it in enumerate(items):
            it["_orig_idx"] = idx
        print(f"[rb2] scoring {len(items)} standard sequences (batch_size={args.batch_size}, "
              f"max_length={args.max_length})", flush=True)
        scores = score_items(items, compute_rewards, tokenizer, device, args.batch_size)
        for it, s in zip(items, scores):
            it["_score"] = s
        standard_results = aggregate(standard_rows, items, tie_epsilon=args.tie_epsilon)

    # ---- Ties subset (separate aggregator) ----
    ties_results: Dict[str, Any] = {}
    if ties_rows:
        ties_items = build_inputs_ties(ties_rows, tokenizer, args.max_length)
        for idx, it in enumerate(ties_items):
            it["_orig_idx"] = idx
        print(f"[rb2] scoring {len(ties_items)} ties candidates "
              f"({len(ties_rows)} rows, variable candidates per row)", flush=True)
        scores = score_items(ties_items, compute_rewards, tokenizer, device, args.batch_size)
        for it, s in zip(ties_items, scores):
            it["_score"] = s
        ties_results = aggregate_ties(ties_rows, ties_items)

    # ---- Combined leaderboard score (mean of 5 subset all_pairs_win + ties overall_score) ----
    combined_components: List[float] = []
    if standard_results:
        for sub in STANDARD_SUBSETS:
            m = standard_results["metrics_per_subset"].get(sub)
            if m and m.get("all_pairs_win_accuracy") is not None:
                combined_components.append(float(m["all_pairs_win_accuracy"]))
    if ties_results:
        ties_overall = ties_results["ties_metrics"].get("overall_score")
        if ties_overall is not None:
            combined_components.append(float(ties_overall))
    combined_score = (sum(combined_components) / len(combined_components)) if combined_components else None

    subsets_evaluated = sorted(
        set(r["subset"] for r in standard_rows) | ({TIES_SUBSET_NAME} if ties_rows else set())
    )

    payload: Dict[str, Any] = {
        "dataset": "allenai/reward-bench-2",
        "base_model": args.base_model,
        "model_path": args.model_path,
        "config": {
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "torch_dtype": args.torch_dtype,
            "subsets_evaluated": subsets_evaluated,
            "skipped_ties": TIES_SUBSET_NAME not in subsets_evaluated,
            "tie_epsilon": args.tie_epsilon,
            "num_examples_capped": args.num_examples,
        },
        "combined_score": combined_score,
        "combined_score_components": {
            "standard_all_pairs_win_per_subset": {
                sub: standard_results["metrics_per_subset"][sub]["all_pairs_win_accuracy"]
                if standard_results and sub in standard_results["metrics_per_subset"] else None
                for sub in STANDARD_SUBSETS
            },
            "ties_overall_score": ties_results["ties_metrics"]["overall_score"] if ties_results else None,
        },
        "metrics_overall": standard_results.get("metrics_overall") if standard_results else None,
        "metrics_per_subset": standard_results.get("metrics_per_subset") if standard_results else None,
        "ties_metrics": ties_results.get("ties_metrics") if ties_results else None,
        "per_example": standard_results.get("per_example", []) if standard_results else [],
        "ties_per_example": ties_results.get("ties_per_example", []) if ties_results else [],
    }
    atomic_write_json(out_path, payload)

    if standard_results:
        mo = standard_results["metrics_overall"]
        print(f"[rb2] STANDARD overall all_pairs_win={mo['all_pairs_win_accuracy']:.4f}  "
              f"pairwise={mo['pairwise_accuracy']:.4f}  n={mo['n_examples']}", flush=True)
        for sub, m in sorted(standard_results["metrics_per_subset"].items()):
            print(f"[rb2]   {sub:18s}  all_pairs_win={m['all_pairs_win_accuracy']:.4f}  "
                  f"pairwise={m['pairwise_accuracy']:.4f}  n={m['n_examples']}", flush=True)
    if ties_results:
        tm = ties_results["ties_metrics"]
        print(f"[rb2] TIES  overall_score={tm['overall_score']:.4f}  "
              f"ref_acc={tm['ref_accuracy']:.4f}  tied_acc={tm['tied_accuracy']:.4f}  "
              f"corr_pref={tm['correctness_preferred']:.4f}  "
              f"corr_pref_hard={tm['correctness_preferred_hard']:.4f}  "
              f"margin_score={tm['correctness_margin_score']:+.4f}  "
              f"n_ref={tm['n_ref_rows']}  n_tied={tm['n_tied_rows']}", flush=True)
    if combined_score is not None:
        print(f"[rb2] COMBINED leaderboard score: {combined_score:.4f}  "
              f"({len(combined_components)} subset scores averaged)  -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
