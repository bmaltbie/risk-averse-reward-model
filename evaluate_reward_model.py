#!/usr/bin/env python3
"""
Evaluate a trained reward model on the risk-averse benchmark.

Instead of generating text, the reward model assigns scalar rewards to each option
in a situation. The option with the highest reward (argmax) is selected as the model's
choice. Output format matches evaluate.py for easy comparison.

Usage:
    python evaluate_reward_model.py --checkpoint_path /path/to/best_checkpoint
    python evaluate_reward_model.py --checkpoint_path /path/to/best_checkpoint --num_situations 100
    python evaluate_reward_model.py --checkpoint_path /path/to/best_checkpoint --cot_csv data/cot.csv
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from eval_utils import (
    remove_instruction_suffix,
    convert_numpy,
    summarize_results,
    build_situations,
)


# Flush output immediately so logs are visible in real time.
sys.stdout.reconfigure(line_buffering=True)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()


# ============================================================
# REWARD MODEL WRAPPER
# ============================================================
class RewardModelWrapper(nn.Module):
    """Lightweight inference wrapper for reward model (LoRA backbone + reward head)."""

    def __init__(self, backbone, reward_head):
        super().__init__()
        self.backbone = backbone
        self.reward_head = reward_head

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        hidden_states = outputs.last_hidden_state
        last_hidden_states = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths,
        ]

        # Convert to fp32 for numerical stability (matches sweep.py)
        last_hidden_states = last_hidden_states.float()
        rewards = self.reward_head(last_hidden_states).squeeze(-1)
        return rewards


def load_reward_model(checkpoint_path: str, base_model_name: str, merge_adapter: bool = False):
    """Load a trained reward model from checkpoint.

    Checkpoint structure (produced by sweep.py):
      - LoRA adapter files (from backbone.save_pretrained)
      - reward_head.pt (reward_head_state_dict, epoch, out_dist_accuracy)

    Returns:
        model: RewardModelWrapper ready for inference
        tokenizer: AutoTokenizer
        checkpoint_info: dict with epoch and accuracy from training
    """
    # Validate checkpoint files exist
    reward_head_path = os.path.join(checkpoint_path, "reward_head.pt")
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")

    if not os.path.exists(reward_head_path):
        raise FileNotFoundError(f"Reward head not found: {reward_head_path}")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"LoRA adapter config not found: {adapter_config_path}")

    print(f"Loading base model: {base_model_name}")
    backbone = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from: {checkpoint_path}")
    backbone = PeftModel.from_pretrained(backbone, checkpoint_path)

    if merge_adapter:
        print("Merging LoRA adapter into base model...")
        backbone = backbone.merge_and_unload()

    # Load reward head
    print(f"Loading reward head from: {reward_head_path}")
    checkpoint_data = torch.load(reward_head_path, map_location="cpu", weights_only=True)
    reward_head_state = checkpoint_data["reward_head_state_dict"]

    hidden_size = backbone.config.hidden_size
    reward_head = nn.Linear(hidden_size, 1, bias=True)
    reward_head.load_state_dict(reward_head_state)

    # Move reward head to same device as backbone, keep in fp32
    device = next(backbone.parameters()).device
    reward_head = reward_head.float().to(device)

    model = RewardModelWrapper(backbone, reward_head)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    checkpoint_info = {
        "epoch": checkpoint_data.get("epoch"),
        "out_dist_accuracy": checkpoint_data.get("out_dist_accuracy"),
    }

    print(f"Model loaded successfully (checkpoint epoch {checkpoint_info['epoch']}, "
          f"val acc {checkpoint_info.get('out_dist_accuracy', 'N/A')})")

    return model, tokenizer, checkpoint_info


# ============================================================
# SCORING
# ============================================================
def score_options_batched(
    model: RewardModelWrapper,
    tokenizer,
    texts: List[str],
    max_length: int = 256,
    batch_size: int = 4,
) -> List[float]:
    """Score a list of texts through the reward model and return scalar rewards."""
    all_rewards = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        device = next(model.parameters()).device
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        try:
            with torch.no_grad():
                rewards = model(input_ids=input_ids, attention_mask=attention_mask)
            all_rewards.extend(rewards.cpu().float().tolist())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  WARNING: GPU OOM during scoring. Clearing cache and retrying one-by-one.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Retry one at a time
                for text in batch_texts:
                    enc = tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    )
                    with torch.no_grad():
                        r = model(
                            input_ids=enc["input_ids"].to(device),
                            attention_mask=enc["attention_mask"].to(device),
                        )
                    all_rewards.append(r.cpu().float().item())
            else:
                raise

    return all_rewards


# ============================================================
# TEXT FORMATTING
# ============================================================
def format_option_text_label(prompt: str, letter: str) -> str:
    """Format option text in label-only mode: prompt + chosen option letter."""
    return f"{prompt}\n\nChosen option: {letter}"


def format_option_text_cot(prompt: str, cot_response: str) -> str:
    """Format option text in CoT mode: prompt + full CoT response."""
    return f"{prompt}\n\n{cot_response}"


# ============================================================
# COT DATA LOADING
# ============================================================
def load_cot_data(cot_csv_path: str) -> Dict[int, Dict[str, str]]:
    """Load CoT CSV and return mapping: {situation_id: {option_letter: cot_text}}.

    The CoT CSV has columns: situation_id, chosen_full, rejected_full,
    chosen_answer, rejected_answer.
    We need to map each option letter to its full CoT text.
    """
    df = pd.read_csv(cot_csv_path)
    cot_map = {}

    for _, row in df.iterrows():
        sit_id = row["situation_id"]
        if sit_id not in cot_map:
            cot_map[sit_id] = {}

        # Map chosen answer letter to its CoT
        if pd.notna(row.get("chosen_full")) and pd.notna(row.get("chosen_answer")):
            letter = str(row["chosen_answer"]).strip().lower()
            cot_map[sit_id][letter] = str(row["chosen_full"])

        # Map rejected answer letter to its CoT
        if pd.notna(row.get("rejected_full")) and pd.notna(row.get("rejected_answer")):
            letter = str(row["rejected_answer"]).strip().lower()
            cot_map[sit_id][letter] = str(row["rejected_full"])

    return cot_map


# ============================================================
# INCREMENTAL SAVE
# ============================================================
def save_incremental(
    output_path: str,
    eval_config: dict,
    results: list,
    failed_responses: list,
    situations_evaluated: int,
):
    """Save current evaluation state to disk for crash resilience."""
    metrics = summarize_results(results)
    valid = [r for r in results if r["option_type"] is not None]

    config_copy = dict(eval_config)
    config_copy["num_situations"] = situations_evaluated

    output_data = convert_numpy(
        {
            "evaluation_config": config_copy,
            "metrics": metrics,
            "num_valid": len(valid),
            "num_total": len(results),
            "results": results,
            "failed_responses": failed_responses[:10],
        }
    )

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================
def run_reward_model_eval(
    model: RewardModelWrapper,
    tokenizer,
    situations: list,
    eval_config: dict,
    output_path: str,
    max_length: int = 256,
    eval_batch_size: int = 4,
    cot_data: Optional[Dict[int, Dict[str, str]]] = None,
):
    """Evaluate reward model on all situations.

    For each situation, scores every option and selects the one with highest reward.
    """
    print(f"Evaluating reward model on {len(situations)} situations...")
    print(f"Max sequence length: {max_length}")
    print(f"Evaluation batch size: {eval_batch_size}")
    print(f"CoT mode: {'YES' if cot_data else 'NO (label-only)'}")
    print(f"Results will be saved incrementally to: {output_path}")
    print()

    results = []
    failed_responses = []
    eval_times = []
    eval_start_time = time.time()

    for i, sit in enumerate(situations):
        sit_start = time.time()
        sit_id = sit["situation_id"]
        prompt = remove_instruction_suffix(sit["prompt_raw"])

        # Build text for each option
        option_letters = sorted(
            set(k for k in sit["options"] if len(k) == 1 and k.isalpha())
        )

        option_texts = []
        for letter in option_letters:
            if cot_data and sit_id in cot_data and letter in cot_data[sit_id]:
                text = format_option_text_cot(prompt, cot_data[sit_id][letter])
            else:
                text = format_option_text_label(prompt, letter)
            option_texts.append(text)

        # Score all options
        rewards = score_options_batched(
            model, tokenizer, option_texts,
            max_length=max_length,
            batch_size=eval_batch_size,
        )

        # Build reward map
        option_rewards = {}
        for letter, reward in zip(option_letters, rewards):
            option_rewards[letter] = round(reward, 6)

        # Select best option (argmax)
        best_idx = max(range(len(rewards)), key=lambda j: rewards[j])
        choice = option_letters[best_idx]
        choice_index = ord(choice) - ord("a") + 1

        # Compute reward margin (best - second best)
        sorted_rewards = sorted(rewards, reverse=True)
        reward_margin = sorted_rewards[0] - sorted_rewards[1] if len(sorted_rewards) > 1 else 0.0

        # Look up option metadata
        if choice in sit["options"]:
            chosen = sit["options"][choice]
            results.append(
                {
                    "situation_id": sit_id,
                    "num_options": sit["num_options"],
                    "probability_format": sit["probability_format"],
                    "bucket_label": sit["bucket_label"],
                    "linear_best_option": sit["linear_best_option"],
                    "cara001_best_option": sit["cara001_best_option"],
                    "choice": choice,
                    "choice_index": choice_index,
                    "parser_strategy": "reward_argmax",
                    "option_type": chosen["type"],
                    "is_best_cara": chosen["is_best_cara"],
                    "is_best_linear": chosen["is_best_linear"],
                    "option_rewards": option_rewards,
                    "reward_margin": round(reward_margin, 6),
                }
            )
        else:
            # Should not happen for reward models, but handle gracefully
            results.append(
                {
                    "situation_id": sit_id,
                    "num_options": sit["num_options"],
                    "probability_format": sit["probability_format"],
                    "bucket_label": sit["bucket_label"],
                    "linear_best_option": sit["linear_best_option"],
                    "cara001_best_option": sit["cara001_best_option"],
                    "choice": choice,
                    "choice_index": choice_index,
                    "parser_strategy": "reward_argmax",
                    "option_type": None,
                    "is_best_cara": None,
                    "is_best_linear": None,
                    "option_rewards": option_rewards,
                    "reward_margin": round(reward_margin, 6),
                }
            )
            failed_responses.append(
                {
                    "situation_id": sit_id,
                    "choice": choice,
                    "reason": f"Choice '{choice}' not found in options",
                }
            )

        sit_elapsed = time.time() - sit_start
        eval_times.append(sit_elapsed)
        avg_time = sum(eval_times) / len(eval_times)
        remaining = avg_time * (len(situations) - i - 1)

        # Format rewards for display
        rewards_str = " ".join(f"{l}={r:.3f}" for l, r in option_rewards.items())
        print(
            f"  [{i+1}/{len(situations)}] sit_id={sit_id} | "
            f"choice={choice} | {rewards_str} | margin={reward_margin:.3f} | "
            f"{sit_elapsed:.1f}s | ETA: {remaining/60:.1f}min"
        )

        # Incremental save
        save_incremental(output_path, eval_config, results, failed_responses, i + 1)

    total_elapsed = time.time() - eval_start_time
    metrics = summarize_results(results)
    valid = [r for r in results if r["option_type"] is not None]

    # Print summary
    print(f"\n{'='*50}")
    print("REWARD MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total situations: {len(situations)}")
    print(f"Valid responses: {len(valid)} ({100*metrics['parse_rate']:.1f}%)")
    print(f"Failed: {len(failed_responses)}")
    print(f"\n% choosing COOPERATE: {100*metrics['cooperate_rate']:.1f}%")
    print(f"% choosing REBEL:     {100*metrics['rebel_rate']:.1f}%")
    print(f"% choosing STEAL:     {100*metrics['steal_rate']:.1f}%")
    print(f"% choosing best CARA: {100*metrics['best_cara_rate']:.1f}%")
    print(f"% choosing best LIN:  {100*metrics['best_linear_rate']:.1f}%")
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes ({total_elapsed:.0f}s)")
    print(f"Avg per situation: {sum(eval_times)/len(eval_times):.1f}s")
    print("=" * 50)

    # Final save
    save_incremental(output_path, eval_config, results, failed_responses, len(situations))
    print(f"\nFinal results saved to {output_path}")

    return metrics


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained reward model on the risk-averse benchmark."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to reward model checkpoint directory (contains adapter_config.json + reward_head.pt)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model ID (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="data/2026-01-29, New merged val set with Rebels and Steals.csv",
        help="Validation CSV file path",
    )
    parser.add_argument(
        "--num_situations",
        type=int,
        default=50,
        help="Number of situations to evaluate (default: 50)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max sequence length for tokenization (default: 256, use 2048 for CoT)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Batch size for scoring (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (auto-generated if omitted)",
    )
    parser.add_argument(
        "--cot_csv",
        type=str,
        default=None,
        help="Optional: CSV with pre-generated CoT responses for each option",
    )
    parser.add_argument(
        "--merge_adapter",
        action="store_true",
        help="Merge LoRA adapter into base model for faster inference",
    )

    args = parser.parse_args()

    # Auto-generate output filename
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_short = os.path.basename(os.path.normpath(args.checkpoint_path))
        parent = os.path.basename(os.path.dirname(os.path.normpath(args.checkpoint_path)))
        if checkpoint_short in ("best_checkpoint", "checkpoint"):
            checkpoint_short = f"{parent}_{checkpoint_short}"
        args.output = f"eval_reward_{checkpoint_short}_{timestamp}.json"

    # Load model
    model, tokenizer, checkpoint_info = load_reward_model(
        args.checkpoint_path,
        args.base_model,
        merge_adapter=args.merge_adapter,
    )

    # Load validation data
    print(f"Loading validation data from: {args.val_csv}")
    df = pd.read_csv(args.val_csv)
    situations = build_situations(df, args.num_situations)
    print(f"Built {len(situations)} situations")

    # Load CoT data if provided
    cot_data = None
    if args.cot_csv:
        print(f"Loading CoT data from: {args.cot_csv}")
        cot_data = load_cot_data(args.cot_csv)
        print(f"Loaded CoT responses for {len(cot_data)} situations")

    # Build evaluation config
    eval_config = {
        "evaluation_type": "reward_model",
        "checkpoint_path": args.checkpoint_path,
        "base_model": args.base_model,
        "val_csv": args.val_csv,
        "num_situations": args.num_situations,
        "max_length": args.max_length,
        "eval_batch_size": args.eval_batch_size,
        "merge_adapter": args.merge_adapter,
        "cot_csv": args.cot_csv,
        "checkpoint_epoch": checkpoint_info.get("epoch"),
        "checkpoint_accuracy": checkpoint_info.get("out_dist_accuracy"),
    }

    # Run evaluation
    metrics = run_reward_model_eval(
        model=model,
        tokenizer=tokenizer,
        situations=situations,
        eval_config=eval_config,
        output_path=args.output,
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size,
        cot_data=cot_data,
    )

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
