#!/usr/bin/env python3
"""
Hyperparameter Sweep Script for Risk-Averse Reward Model Training

Runs multiple training configurations and compares results.
Designed for RunPod or local GPU execution.

Usage:
    python sweep.py --output-dir outputs/sweeps
    python sweep.py --configs 1,3,5  # Run specific configs only
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import sys
import gc
import json
import time
import argparse
import subprocess
import traceback
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class OOMSkipRun(RuntimeError):
    """Raised when a soft OOM should abort the current sweep config and continue with the next."""


def set_seed(seed: int):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class SweepConfig:
    """Configuration for a single training run."""
    name: str
    learning_rate: float = 5e-4
    reward_head_lr_multiplier: float = 1.0  # Reward head LR = learning_rate * this
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    # Fixed params
    model_name: str = "Qwen/Qwen3-8B"
    batch_size: int = 1
    gradient_accumulation_steps: int = 64
    num_epochs: int = 10
    weight_decay: float = 0.05
    max_length: int = 256
    in_dist_val_split: float = 0.10
    random_seed: int = 42
    # CoT training options
    use_cot: bool = True
    cot_max_length: int = 768  # Fits 40GB A100 (actual usage ~515 tokens)
    # Learning rate scheduler
    use_lr_scheduler: bool = True
    warmup_ratio: float = 0.10  # 10% of total steps for warmup
    # Gradient clipping
    use_gradient_clipping: bool = True
    max_grad_norm: float = 1.0  # Standard default for LLM training
    # Reference model for KL regularization
    use_reference_model: bool = True
    reference_kl_beta: float = 0.2  # KL penalty weight (0 = monitoring only)
    # Model initialization
    reward_head_init_std: float = 0.01  # Std dev for reward head weight init
    # Evaluation
    eval_batch_size: int = 4  # Batch size for evaluation (no gradients, can be larger)
    # Early stopping (0 = disabled). Watches out_dist validation accuracy per epoch.
    early_stopping_patience: int = 2
    # OOD validation filtering
    ood_val_num_situations: int = 200  # Limit OOD val set size for faster sweeps
    ood_val_rejected_type: str = "lin"  # Only validate on this rejected_type (None = no filter)
    # Reward model evaluation (post-training, uses evaluate_reward_model.py)
    run_reward_eval: bool = True
    reward_eval_num_situations: int = 50


# Default sweep configurations
# Note: use_cot defaults to True, so label-only configs explicitly set use_cot=False
SWEEP_CONFIGS = [
    # Label-only configurations (for comparison)
    SweepConfig(name="label_lr_5e-5", learning_rate=5e-5, use_cot=False),
    SweepConfig(name="label_lr_1e-4", learning_rate=1e-4, use_cot=False),  # label-only baseline
    # CoT (Chain-of-Thought) configurations - default
    SweepConfig(name="cot_lr_2e-5", learning_rate=2e-5),
    SweepConfig(name="cot_lr_5e-5", learning_rate=5e-5),
    SweepConfig(name="cot_lr_1e-4", learning_rate=1e-4),  # CoT baseline
    SweepConfig(name="cot_lr_2e-4", learning_rate=2e-4),
    SweepConfig(name="cot_lora_r8", lora_r=8, lora_alpha=16),
    SweepConfig(name="cot_lora_r16", lora_r=16, lora_alpha=32),
    SweepConfig(name="cot_lora_r64", lora_r=64, lora_alpha=128),
    # Reward head LR multiplier ablation
    SweepConfig(name="head_mult_1.0x", reward_head_lr_multiplier=1.0),
    SweepConfig(name="head_mult_2.0x", reward_head_lr_multiplier=2.0),
    SweepConfig(name="head_mult_5.0x", reward_head_lr_multiplier=5.0),
]

# Data file paths
# The CoT training file has 1000 situations with both CoT columns and answer labels
COT_TRAIN_DATA_FILE = "data/2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv"
TRAIN_DATA_FILE = COT_TRAIN_DATA_FILE  # Same file; label-only mode drops CoT columns
# Validation files: CoT version for CoT training, cooperate-labels version for label-only
COT_VAL_DATA_FILE = "data/2026_03_22_reward_model_val_set_500_Rebels.csv"
LABEL_VAL_DATA_FILE = "data/2026_03_22_reward_model_val_set_500_Rebels.csv"
VAL_DATA_FILE = COT_VAL_DATA_FILE  # Default (overridden per-config in training function)
DEFAULT_OUTPUT_DIR = "outputs/sweeps"


# ============================================================
# DATA LOADING
# ============================================================
# Schema mapping between training and validation data:
#
# Training data (CARA labels — risk-aversion):
#   correct = CARA_correct_labels (risk-averse choice)
#   incorrect = CARA_alpha_0_10_best_labels | linear_best_labels (based on low_bucket_label)
#   Loaded by: TrainingDataLoader, InDistributionValidationDataLoader
#
# CoT training data (pre-generated chain-of-thought):
#   correct = chosen_full (CoT response with CARA utility → risk-averse answer)
#   incorrect = rejected_full (CoT response with wrong utility function)
#   Loaded by: CoTTrainingDataLoader
#
# Validation data (cooperate labels — out-of-distribution):
#   correct = cooperate_correct_labels (cooperative choice)
#   incorrect = cooperate_incorrect_labels (non-cooperative choice)
#   Loaded by: ValidationDataLoader
#
# Error type classification (why the incorrect option is wrong):
#   too_risky:        preferred by linear (risk-neutral) agent
#   too_risk_averse:  preferred by CARA alpha=0.10 (overly cautious) agent
#   other:            neither pattern


def get_train_val_situation_split(csv_file_path: str, val_fraction: float = 0.10, random_seed: int = 42):
    """Split situation IDs into train and validation sets, stratified by low_bucket_label."""
    df = pd.read_csv(csv_file_path)
    if 'situation_id' not in df.columns:
        raise ValueError(
            f"Missing required column 'situation_id' in '{csv_file_path}'. "
            f"Available: {list(df.columns)}"
        )
    situations = df.groupby('situation_id').first().reset_index()
    situation_ids = situations['situation_id'].tolist()

    if 'low_bucket_label' in situations.columns:
        strat_labels = situations['low_bucket_label'].apply(
            lambda x: x.strip('"') if isinstance(x, str) else x
        ).tolist()
        try:
            train_ids, val_ids = train_test_split(
                situation_ids,
                test_size=val_fraction,
                random_state=random_seed,
                stratify=strat_labels
            )
        except ValueError:
            train_ids, val_ids = train_test_split(
                situation_ids,
                test_size=val_fraction,
                random_state=random_seed
            )
    else:
        train_ids, val_ids = train_test_split(
            situation_ids,
            test_size=val_fraction,
            random_state=random_seed
        )

    return train_ids, val_ids


class TrainingDataLoader:
    """Load and process training data with CARA-based labels."""

    def __init__(self, csv_file_path: str, epoch: int = 0, random_seed: int = 42,
                 situation_ids: List = None):
        self.csv_file_path = csv_file_path
        self.epoch = epoch
        self.rng = np.random.default_rng(random_seed + epoch)
        self.situation_ids = situation_ids

    def load_and_process_data(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"Training data file not found: '{self.csv_file_path}'")
        df = pd.read_csv(self.csv_file_path)

        required_columns = ['situation_id', 'prompt_text', 'CARA_correct_labels', 'low_bucket_label']
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in training data '{self.csv_file_path}': {missing}. "
                f"Available: {list(df.columns)}"
            )
        conditional_columns = ['CARA_alpha_0_10_best_labels', 'linear_best_labels']
        missing_conditional = [c for c in conditional_columns if c not in df.columns]
        if missing_conditional:
            print(f"  Note: Conditional columns not present: {missing_conditional}. "
                  f"Rows with matching low_bucket_labels will be skipped.")

        situations = df.groupby('situation_id').first().reset_index()

        if self.situation_ids is not None:
            situations = situations[situations['situation_id'].isin(self.situation_ids)]

        processed = []
        skipped = 0
        for _, row in situations.iterrows():
            try:
                prompt_text = row['prompt_text']
                correct_labels = json.loads(row['CARA_correct_labels'])
                if not correct_labels:
                    continue

                low_bucket = row['low_bucket_label'].strip('"')

                if low_bucket == '010_only':
                    incorrect_labels = json.loads(row['CARA_alpha_0_10_best_labels'])
                elif low_bucket == 'lin_only':
                    incorrect_labels = json.loads(row['linear_best_labels'])
                elif low_bucket == 'both':
                    if self.rng.random() < 0.5:
                        incorrect_labels = json.loads(row['linear_best_labels'])
                    else:
                        incorrect_labels = json.loads(row['CARA_alpha_0_10_best_labels'])
                else:
                    incorrect_labels = json.loads(row.get('CARA_incorrect_labels', '[]'))

                if not incorrect_labels:
                    continue

                correct_label = str(self.rng.choice(correct_labels))
                incorrect_label = str(self.rng.choice(incorrect_labels))

                processed.append({
                    'situation_id': row['situation_id'],
                    'prompt_text': prompt_text,
                    'correct_label': correct_label,
                    'incorrect_label': incorrect_label,
                    'low_bucket_label': low_bucket,
                })
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                skipped += 1
                continue

        if skipped > 0:
            print(f"  Warning: Skipped {skipped} rows due to parsing errors")
        return pd.DataFrame(processed)


class CoTTrainingDataLoader:
    """Load and process Chain-of-Thought training data.

    This loader works with pre-generated CoT data where each row contains:
    - situation_id: Unique identifier for the situation
    - prompt_text: The decision scenario
    - chosen_full: Full CoT response for the correct answer (includes <think> tags)
    - rejected_full: Full CoT response for the incorrect answer
    - rejected_type: Type of rejection (too_risk, lin, etc.)
    - low_bucket_label: Original bucket label for the situation
    """

    def __init__(self, csv_file_path: str, random_seed: int = 42, situation_ids: List = None):
        self.csv_file_path = csv_file_path
        self.rng = np.random.default_rng(random_seed)
        self.situation_ids = situation_ids

    def load_and_process_data(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"CoT training data file not found: '{self.csv_file_path}'")
        df = pd.read_csv(self.csv_file_path)

        required_columns = ['situation_id', 'prompt_text', 'chosen_full', 'rejected_full']
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in CoT data '{self.csv_file_path}': {missing}. "
                f"Available: {list(df.columns)}"
            )

        if self.situation_ids is not None:
            df = df[df['situation_id'].isin(self.situation_ids)]

        processed = []
        for _, row in df.iterrows():
            try:
                # Map rejected_type to error_type for consistency
                # rejected_type indicates what utility function was used for the rejected option:
                # - 'lin' = linear utility (risk-neutral) → if model picks this, it's being too_risky
                # - 'too_risk' = CARA a=0.10 (overly risk-averse) → if model picks this, it's being too_risk_averse
                # - '010' = same as too_risk (legacy naming)
                rejected_type = row.get('rejected_type', '')
                if rejected_type == 'lin':
                    error_type = 'too_risky'  # linear = risk-neutral = too risky
                elif rejected_type in ('too_risk', '010'):
                    error_type = 'too_risk_averse'  # CARA a=0.10 = overly cautious
                else:
                    error_type = 'other'

                # Skip rows with NaN in critical CoT fields
                chosen = row['chosen_full']
                rejected = row['rejected_full']
                if pd.isna(chosen) or pd.isna(rejected):
                    continue

                processed.append({
                    'situation_id': row['situation_id'],
                    'prompt_text': row['prompt_text'],
                    'chosen_full': chosen,
                    'rejected_full': rejected,
                    'correct_label': row.get('chosen_answer', ''),
                    'incorrect_label': row.get('rejected_answer', ''),
                    'error_type': error_type,
                    'low_bucket_label': row.get('low_bucket_label', ''),
                })
            except KeyError as e:
                print(f"Warning: Missing key {e} in row {row.get('situation_id', 'unknown')}")
                continue

        return pd.DataFrame(processed)


class InDistributionValidationDataLoader:
    """Load in-distribution validation data with CARA-based labels."""

    def __init__(self, csv_file_path: str, random_seed: int = 42, situation_ids: List = None):
        self.csv_file_path = csv_file_path
        self.rng = np.random.default_rng(random_seed)
        self.situation_ids = situation_ids

    def load_and_process_data(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"In-distribution validation data file not found: '{self.csv_file_path}'")
        df = pd.read_csv(self.csv_file_path)

        required_columns = ['situation_id', 'prompt_text', 'CARA_correct_labels', 'low_bucket_label']
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in in-dist validation data '{self.csv_file_path}': {missing}. "
                f"Available: {list(df.columns)}"
            )
        conditional_columns = ['CARA_alpha_0_10_best_labels', 'linear_best_labels']
        missing_conditional = [c for c in conditional_columns if c not in df.columns]
        if missing_conditional:
            print(f"  Note: Conditional columns not present: {missing_conditional}. "
                  f"Rows with matching low_bucket_labels will be skipped.")

        situations = df.groupby('situation_id').first().reset_index()

        if self.situation_ids is not None:
            situations = situations[situations['situation_id'].isin(self.situation_ids)]

        processed = []
        skipped = 0
        for _, row in situations.iterrows():
            try:
                prompt_text = row['prompt_text']
                correct_labels = json.loads(row['CARA_correct_labels'])
                if not correct_labels:
                    continue

                low_bucket = row['low_bucket_label'].strip('"')

                if low_bucket == '010_only':
                    incorrect_labels = json.loads(row['CARA_alpha_0_10_best_labels'])
                    error_type = 'too_risk_averse'
                elif low_bucket == 'lin_only':
                    incorrect_labels = json.loads(row['linear_best_labels'])
                    error_type = 'too_risky'
                elif low_bucket == 'both':
                    if self.rng.random() < 0.5:
                        incorrect_labels = json.loads(row['linear_best_labels'])
                        error_type = 'too_risky'
                    else:
                        incorrect_labels = json.loads(row['CARA_alpha_0_10_best_labels'])
                        error_type = 'too_risk_averse'
                else:
                    incorrect_labels = json.loads(row.get('CARA_incorrect_labels', '[]'))
                    error_type = 'other'

                if not incorrect_labels:
                    continue

                correct_label = str(self.rng.choice(correct_labels))
                incorrect_label = str(self.rng.choice(incorrect_labels))

                processed.append({
                    'situation_id': row['situation_id'],
                    'prompt_text': prompt_text,
                    'correct_label': correct_label,
                    'incorrect_label': incorrect_label,
                    'error_type': error_type,
                    'low_bucket_label': low_bucket,
                })
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                skipped += 1
                continue

        if skipped > 0:
            print(f"  Warning: Skipped {skipped} rows due to parsing errors")
        return pd.DataFrame(processed)


def _is_true(value) -> bool:
    """Check if a value is TRUE (handles NaN as FALSE)."""
    if pd.isna(value):
        return False
    return str(value).upper() == 'TRUE'


class ValidationDataLoader:
    """Load validation data with cooperate-based labels."""

    def __init__(self, csv_file_path: str, random_seed: int = 42):
        self.csv_file_path = csv_file_path
        self.rng = np.random.default_rng(random_seed)

    def load_and_process_data(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"Validation data file not found: '{self.csv_file_path}'")
        df = pd.read_csv(self.csv_file_path)
        df = df.dropna(how='all')

        required_columns = ['situation_id', 'prompt_text', 'cooperate_correct_labels',
                            'cooperate_incorrect_labels', 'option_index']
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in validation data '{self.csv_file_path}': {missing}. "
                f"Available: {list(df.columns)}"
            )

        # Build option properties lookup
        option_properties = {}
        for _, row in df.iterrows():
            sit_id = row['situation_id']
            opt_idx = int(row['option_index'])

            correct_labels_str = row['cooperate_correct_labels']
            if correct_labels_str and pd.notna(correct_labels_str):
                try:
                    sample_labels = json.loads(correct_labels_str)
                    if sample_labels and str(sample_labels[0]).isdigit():
                        label = str(opt_idx + 1)
                    else:
                        label = chr(ord('a') + opt_idx)
                except json.JSONDecodeError:
                    label = chr(ord('a') + opt_idx)
            else:
                label = chr(ord('a') + opt_idx)

            is_linear_best = _is_true(row.get('is_best_linear_display'))
            alpha_010_str = row.get('CARA_alpha_0_10_best_labels', '')
            alpha_010_labels = []
            if alpha_010_str and pd.notna(alpha_010_str) and str(alpha_010_str).strip():
                try:
                    alpha_010_labels = json.loads(alpha_010_str)
                except json.JSONDecodeError:
                    pass

            option_properties[(sit_id, label)] = {
                'is_linear_best': is_linear_best,
                'alpha_010_labels': alpha_010_labels,
            }

        situations = df.groupby('situation_id').first().reset_index()

        processed = []
        skipped = 0
        for _, row in situations.iterrows():
            try:
                sit_id = row['situation_id']

                if pd.isna(row['cooperate_correct_labels']) or pd.isna(row['cooperate_incorrect_labels']):
                    continue

                correct_labels = json.loads(row['cooperate_correct_labels'])
                incorrect_labels = json.loads(row['cooperate_incorrect_labels'])

                if not correct_labels or not incorrect_labels:
                    continue

                correct_label = str(self.rng.choice(correct_labels))
                incorrect_label = str(self.rng.choice(incorrect_labels))

                props = option_properties.get((sit_id, incorrect_label), {})
                is_linear_best = props.get('is_linear_best', False)
                alpha_010_labels = props.get('alpha_010_labels', [])

                if is_linear_best:
                    error_type = 'too_risky'
                elif incorrect_label in alpha_010_labels:
                    error_type = 'too_risk_averse'
                else:
                    error_type = 'other'

                processed.append({
                    'situation_id': sit_id,
                    'prompt_text': row['prompt_text'],
                    'correct_label': correct_label,
                    'incorrect_label': incorrect_label,
                    'error_type': error_type,
                })
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                skipped += 1
                continue

        if skipped > 0:
            print(f"  Warning: Skipped {skipped} rows due to parsing errors")
        return pd.DataFrame(processed)


# ============================================================
# DATASET
# ============================================================
class PairwiseRewardDataset(Dataset):
    """Dataset for pairwise reward model training.

    Supports two formats:
    1. Label-only: Uses prompt_text + correct_label/incorrect_label to construct pairs
    2. CoT (Chain-of-Thought): Uses chosen_full/rejected_full columns directly

    The format is auto-detected based on column presence.
    """

    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = 256):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_error_types = 'error_type' in self.data.columns
        # Auto-detect CoT format by checking for chosen_full/rejected_full columns
        self.use_cot = 'chosen_full' in self.data.columns and 'rejected_full' in self.data.columns
        if self.has_error_types:
            self.error_types = self.data['error_type'].tolist()
        else:
            self.error_types = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if self.use_cot:
            # CoT format: use full chain-of-thought responses
            # Format: "{prompt}\n\n{cot_response}" where cot_response includes <think> tags
            preferred_text = f"{row['prompt_text']}\n\n{row['chosen_full']}"
            rejected_text = f"{row['prompt_text']}\n\n{row['rejected_full']}"
        else:
            # Label-only format: construct from prompt + label
            preferred_text = f"{row['prompt_text']}\n\nChosen option: {row['correct_label']}"
            rejected_text = f"{row['prompt_text']}\n\nChosen option: {row['incorrect_label']}"

        preferred_encoding = self.tokenizer(
            preferred_text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        rejected_encoding = self.tokenizer(
            rejected_text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )

        result = {
            'preferred_input_ids': preferred_encoding['input_ids'].squeeze(0),
            'preferred_attention_mask': preferred_encoding['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(0),
        }

        if self.has_error_types:
            result['error_type'] = self.error_types[idx]

        return result


# ============================================================
# MODEL
# ============================================================
class RewardModel(nn.Module):
    """Reward model with LoRA-adapted backbone and trainable reward head."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None,
        reward_head_init_std: float = 0.01,
    ):
        super().__init__()

        print(f"Loading base model: {model_name}")
        self.backbone = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
        )

        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1, bias=True)
        nn.init.normal_(self.reward_head.weight, mean=0.0, std=reward_head_init_std)
        nn.init.zeros_(self.reward_head.bias)

        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total trainable: {trainable_params:,} / {total_params:,}")

    def _extract_last_hidden(self, input_ids, attention_mask):
        """Run backbone and extract last-token hidden states in fp32."""
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
            sequence_lengths
        ]

        return last_hidden_states.float()

    def forward(self, input_ids, attention_mask, return_hidden=False):
        last_hidden_states = self._extract_last_hidden(input_ids, attention_mask)
        rewards = self.reward_head(last_hidden_states).squeeze(-1)

        if return_hidden:
            return rewards, last_hidden_states
        return rewards


# ============================================================
# LOSS AND EVALUATION
# ============================================================
def bradley_terry_loss(preferred_rewards, rejected_rewards):
    """Compute Bradley-Terry pairwise ranking loss."""
    return -F.logsigmoid(preferred_rewards - rejected_rewards).mean()


def compute_reward_kl(pref, rej, ref_pref, ref_rej):
    """Compute KL divergence between trained and reference reward distributions.

    Treats the pair of rewards as a 2-class distribution and computes
    KL(trained || reference) to penalize divergence from the reference.
    """
    # Stack rewards into logits for 2-class distribution
    trained_logits = torch.stack([pref, rej], dim=-1)
    ref_logits = torch.stack([ref_pref, ref_rej], dim=-1)

    # KL(trained || reference)
    trained_log_probs = F.log_softmax(trained_logits, dim=-1)
    ref_probs = F.softmax(ref_logits, dim=-1)

    return F.kl_div(trained_log_probs, ref_probs, reduction='batchmean')


def compute_reference_rewards(reference_head, last_hidden_states):
    """Compute rewards using frozen reference head from cached hidden states."""
    with torch.no_grad():
        return reference_head(last_hidden_states.detach()).squeeze(-1)


def evaluate_model(model, dataset, device, batch_size=4, verbose=True, desc="eval"):
    """Evaluate model on pairwise accuracy.

    When verbose=True, prints a progress line every ~10% of batches so that long
    eval passes (e.g. baseline on the full val set with the 8B backbone) are not silent.
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_batches = len(dataloader)
    log_every = max(1, n_batches // 10)
    start_t = time.time()
    oom_skipped = 0
    if verbose:
        print(f"  [{desc}] {len(dataset)} pairs in {n_batches} batches (bs={batch_size})...", flush=True)

    correct = 0
    total = 0
    total_loss = 0.0
    preferred_scores = []
    rejected_scores = []
    correct_by_type = {'too_risky': 0, 'too_risk_averse': 0, 'other': 0}
    total_by_type = {'too_risky': 0, 'too_risk_averse': 0, 'other': 0}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_ok = True
            try:
                preferred_rewards = model(
                    input_ids=batch['preferred_input_ids'].to(device),
                    attention_mask=batch['preferred_attention_mask'].to(device)
                )
                rejected_rewards = model(
                    input_ids=batch['rejected_input_ids'].to(device),
                    attention_mask=batch['rejected_attention_mask'].to(device)
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                msg = str(e).lower()
                if "out of memory" in msg or "cublas_status_alloc_failed" in msg or isinstance(e, torch.cuda.OutOfMemoryError):
                    print(f"  [{desc}] WARNING: GPU OOM on batch {batch_idx + 1}/{n_batches}. "
                          f"Aborting this run.", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise OOMSkipRun(f"OOM during {desc} batch {batch_idx + 1}") from e
                else:
                    raise

            if batch_ok:
                loss = bradley_terry_loss(preferred_rewards, rejected_rewards)
                total_loss += loss.item() * len(preferred_rewards)

                is_correct = (preferred_rewards > rejected_rewards)
                correct += is_correct.sum().item()
                total += len(preferred_rewards)

                preferred_scores.extend(preferred_rewards.cpu().float().numpy().tolist())
                rejected_scores.extend(rejected_rewards.cpu().float().numpy().tolist())

                if 'error_type' in batch:
                    for i, error_type in enumerate(batch['error_type']):
                        if error_type in total_by_type:
                            total_by_type[error_type] += 1
                            if is_correct[i].item():
                                correct_by_type[error_type] += 1

            # Progress print runs for EVERY batch outcome (ok or OOM-skipped), so
            # a silent hang is distinguishable from a storm of OOMs. First 3 batches
            # print unconditionally to confirm liveness; after that, every ~10%.
            if verbose and (batch_idx < 3 or (batch_idx + 1) % log_every == 0 or batch_idx == n_batches - 1):
                elapsed = time.time() - start_t
                running_acc = correct / max(total, 1)
                pct = 100.0 * (batch_idx + 1) / n_batches
                oom_note = f" oom_skipped={oom_skipped}" if oom_skipped else ""
                print(f"  [{desc}] batch {batch_idx + 1}/{n_batches} ({pct:5.1f}%) "
                      f"acc={running_acc:.3f} elapsed={elapsed:.1f}s{oom_note}", flush=True)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    if verbose:
        oom_note = f" (OOM-skipped {oom_skipped}/{n_batches} batches)" if oom_skipped else ""
        print(f"  [{desc}] done in {time.time() - start_t:.1f}s — "
              f"final acc={accuracy:.4f} loss={avg_loss:.4f}{oom_note}", flush=True)

    error_type_breakdown = {}
    for error_type in ['too_risky', 'too_risk_averse', 'other']:
        if total_by_type[error_type] > 0:
            error_type_breakdown[error_type] = {
                'accuracy': correct_by_type[error_type] / total_by_type[error_type],
                'correct': correct_by_type[error_type],
                'total': total_by_type[error_type],
            }

    model.train()

    return {
        'accuracy': accuracy,
        'avg_loss': avg_loss,
        'preferred_scores': preferred_scores,
        'rejected_scores': rejected_scores,
        'error_type_breakdown': error_type_breakdown,
    }


# ============================================================
# GENERATIVE EVALUATION
# ============================================================
# Reward model eval script (upstream submodule) and data
# We rely on eval/risk-averse-ai-eval/evaluate_reward_model.py for pairwise
# scoring of (chosen_full, rejected_full) CoT transcripts.
#
# We run TWO evals per checkpoint:
#   - clean_400: upstream's canonical 400-row alias (audited; externally comparable)
#   - rebels_500: the raw 500-row file (includes 82 audit-flagged rows; absolute
#                 score deflated by ~10-15pts but useful for tracking how well a
#                 model handles noisy/ambiguous "preferred" CoTs)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_SCRIPT = os.path.join(_PROJECT_ROOT, "eval", "risk-averse-ai-eval", "evaluate_reward_model.py")
EVAL_VAL_CSV_500 = os.path.join(_PROJECT_ROOT, "data", "2026_03_22_reward_model_val_set_500_Rebels.csv")
# Each spec: (key, label, csv_path_or_None, dataset_alias_or_None)
# csv_path takes precedence; dataset_alias is resolved by the upstream script.
REWARD_EVAL_SPECS = [
    ("clean_400", "400 (clean)", None, "reward_model_validation"),
    ("rebels_500", "500 (raw)", EVAL_VAL_CSV_500, None),
]


def _run_single_reward_eval(
    config: SweepConfig,
    checkpoint_dir: str,
    output_dir: str,
    eval_key: str,
    eval_label: str,
    csv_path: Optional[str],
    dataset_alias: Optional[str],
    max_length: int,
) -> Optional[dict]:
    """Invoke upstream evaluate_reward_model.py once and parse its summary."""
    output_json = os.path.join(output_dir, f"reward_eval_results_{eval_key}.json")

    # Upstream CLI:
    #   --model_path  : PEFT adapter path (LoRA + reward head checkpoint)
    #   --base_model  : base HF model id
    #   --custom_csv  : pairwise CoT CSV (prompt_text/chosen_full/rejected_full)
    #   --dataset     : built-in alias (resolved against submodule's data/ dir)
    #   --num_pairs   : cap on dedup'd pair rows to score
    #   --batch_size  : pairs scored in parallel
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--model_path", checkpoint_dir,
        "--base_model", config.model_name,
        "--num_pairs", str(config.reward_eval_num_situations),
        "--max_length", str(max_length),
        "--batch_size", str(config.eval_batch_size),
        "--output", output_json,
    ]
    if csv_path:
        cmd += ["--custom_csv", csv_path]
        dataset_descr = os.path.basename(csv_path)
    else:
        cmd += ["--dataset", dataset_alias]
        dataset_descr = dataset_alias

    print(f"\n  [{eval_label}] running ({config.reward_eval_num_situations} pairs, max_length={max_length}, dataset={dataset_descr})...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode != 0:
            print(f"  [{eval_label}] WARNING: failed (exit code {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[-20:]:
                    print(f"    {line}")
            return None

        if not os.path.isfile(output_json):
            print(f"  [{eval_label}] WARNING: produced no output file")
            return None

        with open(output_json, 'r') as f:
            eval_results = json.load(f)

        metrics = eval_results.get('metrics', {})
        summary = {
            'pairwise_accuracy': metrics.get('pairwise_accuracy', None),
            'pairwise_accuracy_ties_half_credit': metrics.get('pairwise_accuracy_ties_half_credit', None),
            'tie_rate': metrics.get('tie_rate', None),
            'preference_log_loss': metrics.get('preference_log_loss', None),
            'mean_score_margin': metrics.get('mean_score_margin', None),
            'mean_accepted_score': metrics.get('mean_accepted_score', None),
            'mean_rejected_score': metrics.get('mean_rejected_score', None),
            'truncated_pair_rate': metrics.get('truncated_pair_rate', None),
            'num_pairs': eval_results.get('num_total', config.reward_eval_num_situations),
            'num_correct': eval_results.get('num_correct', None),
            'num_ties': eval_results.get('num_ties', 0) or 0,
            'eval_dataset': dataset_descr,
        }

        acc = summary['pairwise_accuracy']
        margin = summary['mean_score_margin']
        tie = summary['tie_rate']
        print(f"  [{eval_label}] pairwise_acc={acc:.3f}" if acc is not None else f"  [{eval_label}] pairwise_acc=N/A",
              end="")
        print(f"  margin={margin:+.4f}" if margin is not None else "  margin=N/A", end="")
        print(f"  tie_rate={tie:.3f}" if tie is not None else "  tie_rate=N/A", end="")
        print(f"  ({summary['num_correct']}/{summary['num_pairs']} correct)")

        return summary

    except subprocess.TimeoutExpired:
        print(f"  [{eval_label}] WARNING: timed out after 30 minutes")
        return None
    except Exception as e:
        print(f"  [{eval_label}] WARNING: error: {e}")
        return None


def run_reward_model_evaluation(config: SweepConfig, checkpoint_dir: str, output_dir: str, device: torch.device) -> Optional[dict]:
    """Run upstream reward-model eval against every spec in REWARD_EVAL_SPECS.

    Returns a dict keyed by spec name (e.g. {"clean_400": {...}, "rebels_500": {...}})
    where each value is the per-eval summary. Returns None if no eval ran successfully.
    """
    if not os.path.isfile(EVAL_SCRIPT):
        print(f"  WARNING: Reward eval skipped - eval script not found at {EVAL_SCRIPT}")
        print(f"  To enable: git submodule update --init --recursive")
        return None
    if not os.path.isdir(checkpoint_dir):
        print(f"  WARNING: Reward eval skipped - checkpoint not found at {checkpoint_dir}")
        return None

    max_length = config.cot_max_length if config.use_cot else config.max_length

    aggregated = {}
    for key, label, csv_path, dataset_alias in REWARD_EVAL_SPECS:
        if csv_path and not os.path.isfile(csv_path):
            print(f"  [{label}] skipped - CSV not found at {csv_path}")
            continue
        summary = _run_single_reward_eval(
            config, checkpoint_dir, output_dir, key, label, csv_path, dataset_alias, max_length,
        )
        if summary is not None:
            aggregated[key] = summary

    return aggregated or None


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_single_run(config: SweepConfig, output_dir: str, device: torch.device) -> dict:
    """Run training with given config, return results dict."""
    set_seed(config.random_seed)
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training: {config.name}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}")
    print(f"{'='*60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine max_length based on CoT setting
    if config.use_cot:
        train_data_file = COT_TRAIN_DATA_FILE
        train_max_length = config.cot_max_length
        out_dist_val_file = COT_VAL_DATA_FILE
        print(f"  Using CoT training data: {train_data_file}")
        print(f"  CoT max_length: {train_max_length}")
    else:
        train_data_file = TRAIN_DATA_FILE
        train_max_length = config.max_length
        out_dist_val_file = LABEL_VAL_DATA_FILE
        print(f"  Using label-only training (CoT columns dropped)")
        print(f"  Label max_length: {train_max_length}")

    # Load data with train/in-dist-val split
    train_ids, in_dist_val_ids = get_train_val_situation_split(
        train_data_file,
        val_fraction=config.in_dist_val_split,
        random_seed=config.random_seed
    )

    # Both CoT and label-only modes load from the CoT file via CoTTrainingDataLoader
    # (it extracts correct_label/incorrect_label from chosen_answer/rejected_answer).
    # For label-only mode, we drop the CoT columns so PairwiseRewardDataset uses labels.
    train_loader = CoTTrainingDataLoader(
        train_data_file, random_seed=config.random_seed, situation_ids=train_ids
    )
    train_df = train_loader.load_and_process_data()
    if train_df.empty:
        raise ValueError(f"Training data is empty after processing '{train_data_file}'. Check data file and filters.")

    in_dist_val_loader = CoTTrainingDataLoader(
        train_data_file, random_seed=config.random_seed, situation_ids=in_dist_val_ids
    )
    in_dist_val_df = in_dist_val_loader.load_and_process_data()
    if in_dist_val_df.empty:
        raise ValueError(f"In-distribution validation data is empty after processing. Check data file and situation ID split.")

    # Out-dist validation: always use CoTTrainingDataLoader since val set is in pairwise CoT format
    out_dist_val_loader = CoTTrainingDataLoader(out_dist_val_file, random_seed=config.random_seed)
    out_dist_val_df = out_dist_val_loader.load_and_process_data()
    if out_dist_val_df.empty:
        raise ValueError(f"Out-of-distribution validation data is empty after processing '{out_dist_val_file}'. Check data file.")

    # Filter OOD val by rejected_type and limit number of situations for faster sweeps
    if config.ood_val_rejected_type and 'error_type' in out_dist_val_df.columns:
        # error_type mapping: 'lin' rejected_type → 'too_risky' error_type, 'too_risk' → 'too_risk_averse'
        target_error_type = 'too_risky' if config.ood_val_rejected_type == 'lin' else 'too_risk_averse'
        before = len(out_dist_val_df)
        out_dist_val_df = out_dist_val_df[out_dist_val_df['error_type'] == target_error_type].reset_index(drop=True)
        print(f"  OOD val filtered to error_type='{target_error_type}': {before} → {len(out_dist_val_df)} pairs")
    if config.ood_val_num_situations and config.ood_val_num_situations < len(out_dist_val_df):
        rng = np.random.default_rng(config.random_seed)
        sit_ids = out_dist_val_df['situation_id'].unique()
        chosen_ids = rng.choice(sit_ids, size=config.ood_val_num_situations, replace=False)
        out_dist_val_df = out_dist_val_df[out_dist_val_df['situation_id'].isin(chosen_ids)].reset_index(drop=True)
        print(f"  OOD val limited to {config.ood_val_num_situations} situations: {len(out_dist_val_df)} pairs")

    # For label-only mode, drop CoT columns so PairwiseRewardDataset uses label format
    if not config.use_cot:
        for col in ['chosen_full', 'rejected_full']:
            if col in train_df.columns:
                train_df = train_df.drop(columns=[col])
            if col in in_dist_val_df.columns:
                in_dist_val_df = in_dist_val_df.drop(columns=[col])
            if col in out_dist_val_df.columns:
                out_dist_val_df = out_dist_val_df.drop(columns=[col])

    train_dataset = PairwiseRewardDataset(train_df, tokenizer, max_length=train_max_length)
    in_dist_val_dataset = PairwiseRewardDataset(in_dist_val_df, tokenizer, max_length=train_max_length)
    out_dist_val_dataset = PairwiseRewardDataset(out_dist_val_df, tokenizer, max_length=train_max_length)

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  In-dist validation: {len(in_dist_val_dataset)}")
    print(f"  Out-dist validation: {len(out_dist_val_dataset)}")

    # Create model
    model = RewardModel(
        model_name=config.model_name,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        reward_head_init_std=config.reward_head_init_std,
    )

    # Ensure reward head is on device and fp32
    model.reward_head = model.reward_head.to(device).float()

    # Create frozen reference head for KL regularization
    reference_reward_head = None
    if config.use_reference_model:
        hidden_size = model.backbone.config.hidden_size
        reference_reward_head = nn.Linear(hidden_size, 1, bias=True)
        reference_reward_head.load_state_dict(model.reward_head.state_dict())
        reference_reward_head = reference_reward_head.to(device).float()
        reference_reward_head.eval()
        for param in reference_reward_head.parameters():
            param.requires_grad = False
        print(f"Reference reward head created (frozen, {sum(p.numel() for p in reference_reward_head.parameters())} params)")

    # Create optimizer with parameter groups
    lora_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    reward_head_lr = config.learning_rate * config.reward_head_lr_multiplier
    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': config.learning_rate},
        {'params': model.reward_head.parameters(), 'lr': reward_head_lr},
    ], weight_decay=config.weight_decay)
    print(f"Optimizer: LoRA LR={config.learning_rate}, Reward Head LR={reward_head_lr} ({config.reward_head_lr_multiplier}x)")

    # Calculate total training steps for scheduler
    num_train_examples = len(train_dataset)
    steps_per_epoch = (num_train_examples + config.batch_size - 1) // config.batch_size
    effective_steps_per_epoch = (steps_per_epoch + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps
    total_training_steps = effective_steps_per_epoch * config.num_epochs

    # Create learning rate scheduler (cosine with warmup)
    scheduler = None
    if config.use_lr_scheduler:
        warmup_steps = int(config.warmup_ratio * total_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
        print(f"\nLR Scheduler: cosine with {warmup_steps} warmup steps, {total_training_steps} total steps")

    if config.use_gradient_clipping:
        print(f"Gradient Clipping: max_norm={config.max_grad_norm}")

    if config.use_reference_model:
        print(f"Reference Model: KL beta={config.reference_kl_beta}")

    # Baseline evaluation
    print("\nBaseline evaluation (untrained reward head)...")
    eval_bs = config.eval_batch_size
    baseline_out = evaluate_model(model, out_dist_val_dataset, device, batch_size=eval_bs,
                                  verbose=True, desc="baseline out-dist")
    baseline_in = evaluate_model(model, in_dist_val_dataset, device, batch_size=eval_bs,
                                 verbose=True, desc="baseline in-dist")
    print(f"  Baseline out-dist accuracy: {baseline_out['accuracy']:.4f}")
    print(f"  Baseline in-dist accuracy: {baseline_in['accuracy']:.4f}")

    # Training history
    history = {
        'train_loss': [],
        'kl_loss': [],
        'in_dist_val_accuracy': [],
        'out_dist_val_accuracy': [],
        'epochs': [],
        # Reference model tracking
        'trained_pref_reward_mean': [],
        'trained_rej_reward_mean': [],
        'reference_pref_reward_mean': [],
        'reference_rej_reward_mean': [],
        'reward_divergence': [],  # |trained_margin - reference_margin|
    }

    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0  # For early stopping

    # Training loop
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Note: Training data is pre-generated (CoT pairs with fixed chosen/rejected),
        # so no per-epoch re-randomization is needed for either CoT or label-only mode.

        train_dataloader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            pin_memory=torch.cuda.is_available()
        )

        model.train()
        epoch_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_trained_pref = []
        epoch_trained_rej = []
        epoch_ref_pref = []
        epoch_ref_rej = []
        num_batches = len(train_dataloader)

        optimizer.zero_grad()
        oom_encountered = False
        for step, batch in enumerate(train_dataloader):
            try:
                preferred_input_ids = batch['preferred_input_ids'].to(device)
                preferred_attention_mask = batch['preferred_attention_mask'].to(device)
                rejected_input_ids = batch['rejected_input_ids'].to(device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(device)

                # Use return_hidden when we need reference rewards to avoid redundant backbone pass
                need_hidden = reference_reward_head is not None and config.reference_kl_beta > 0
                if need_hidden:
                    preferred_rewards, pref_hidden = model(
                        input_ids=preferred_input_ids,
                        attention_mask=preferred_attention_mask,
                        return_hidden=True
                    )
                    rejected_rewards, rej_hidden = model(
                        input_ids=rejected_input_ids,
                        attention_mask=rejected_attention_mask,
                        return_hidden=True
                    )
                else:
                    preferred_rewards = model(
                        input_ids=preferred_input_ids,
                        attention_mask=preferred_attention_mask
                    )
                    rejected_rewards = model(
                        input_ids=rejected_input_ids,
                        attention_mask=rejected_attention_mask
                    )

                loss = bradley_terry_loss(preferred_rewards, rejected_rewards)

                # Add KL regularization if reference model is enabled
                kl_loss = torch.tensor(0.0, device=device)
                if need_hidden:
                    ref_pref_rewards = compute_reference_rewards(reference_reward_head, pref_hidden)
                    ref_rej_rewards = compute_reference_rewards(reference_reward_head, rej_hidden)

                    kl_loss = compute_reward_kl(
                        preferred_rewards, rejected_rewards,
                        ref_pref_rewards, ref_rej_rewards
                    )
                    loss = loss + config.reference_kl_beta * kl_loss

                scaled_loss = loss / config.gradient_accumulation_steps

                if torch.isnan(loss):
                    print(f"  WARNING: NaN loss at epoch {epoch+1}, step {step+1}. Zeroing gradients and skipping.")
                    optimizer.zero_grad()
                    continue

                # Track rewards AFTER NaN check to avoid polluting history
                epoch_trained_pref.extend(preferred_rewards.detach().cpu().tolist())
                epoch_trained_rej.extend(rejected_rewards.detach().cpu().tolist())
                epoch_kl_loss += kl_loss.item()
                if need_hidden:
                    epoch_ref_pref.extend(ref_pref_rewards.detach().cpu().tolist())
                    epoch_ref_rej.extend(ref_rej_rewards.detach().cpu().tolist())

                scaled_loss.backward()
                epoch_loss += loss.item()

                is_accumulation_complete = (step + 1) % config.gradient_accumulation_steps == 0
                is_last_batch = (step + 1) == num_batches

                if is_accumulation_complete or is_last_batch:
                    if config.use_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  WARNING: GPU OOM at epoch {epoch+1}, step {step+1}. Aborting this run.")
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise OOMSkipRun(f"OOM at epoch {epoch+1}, step {step+1}") from e
                raise

        if oom_encountered:
            print(f"  Note: OOM occurred during epoch {epoch+1}. Consider reducing batch_size or max_length.")

        avg_train_loss = epoch_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches if epoch_kl_loss > 0 else 0.0

        # Validation
        in_dist_eval = evaluate_model(model, in_dist_val_dataset, device, batch_size=eval_bs)
        out_dist_eval = evaluate_model(model, out_dist_val_dataset, device, batch_size=eval_bs)

        print(f"  Loss: {avg_train_loss:.4f}")
        if config.use_reference_model and config.reference_kl_beta > 0:
            print(f"  KL Loss: {avg_kl_loss:.4f}")
        print(f"  In-dist accuracy: {in_dist_eval['accuracy']:.4f}")
        print(f"  Out-dist accuracy: {out_dist_eval['accuracy']:.4f}")

        history['train_loss'].append(avg_train_loss)
        history['kl_loss'].append(avg_kl_loss)
        history['in_dist_val_accuracy'].append(in_dist_eval['accuracy'])
        history['out_dist_val_accuracy'].append(out_dist_eval['accuracy'])
        history['epochs'].append(epoch + 1)

        # Track reward statistics for visualization
        trained_pref_mean = np.mean(epoch_trained_pref) if epoch_trained_pref else 0.0
        trained_rej_mean = np.mean(epoch_trained_rej) if epoch_trained_rej else 0.0
        history['trained_pref_reward_mean'].append(trained_pref_mean)
        history['trained_rej_reward_mean'].append(trained_rej_mean)

        if epoch_ref_pref:
            ref_pref_mean = np.mean(epoch_ref_pref)
            ref_rej_mean = np.mean(epoch_ref_rej)
            trained_margin = trained_pref_mean - trained_rej_mean
            ref_margin = ref_pref_mean - ref_rej_mean
            divergence = abs(trained_margin - ref_margin)
        else:
            ref_pref_mean = 0.0
            ref_rej_mean = 0.0
            divergence = 0.0
        history['reference_pref_reward_mean'].append(ref_pref_mean)
        history['reference_rej_reward_mean'].append(ref_rej_mean)
        history['reward_divergence'].append(divergence)

        if out_dist_eval['accuracy'] > best_val_accuracy:
            best_val_accuracy = out_dist_eval['accuracy']
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            # Save best checkpoint
            checkpoint_dir = os.path.join(output_dir, 'best_checkpoint')
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.backbone.save_pretrained(checkpoint_dir)
            torch.save({
                'reward_head_state_dict': model.reward_head.state_dict(),
                'epoch': best_epoch,
                'out_dist_accuracy': best_val_accuracy,
            }, os.path.join(checkpoint_dir, 'reward_head.pt'))
            print(f"  New best! Saved checkpoint (epoch {best_epoch}, acc={best_val_accuracy:.4f})")
        else:
            epochs_without_improvement += 1
            if config.early_stopping_patience and epochs_without_improvement >= config.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch + 1} "
                      f"(no improvement for {config.early_stopping_patience} epochs; "
                      f"best={best_val_accuracy:.4f} @ epoch {best_epoch})")
                break

    # Final evaluation
    final_out_eval = evaluate_model(model, out_dist_val_dataset, device, batch_size=eval_bs)
    final_in_eval = evaluate_model(model, in_dist_val_dataset, device, batch_size=eval_bs)

    elapsed = time.time() - start_time

    # Build results
    results = {
        'config': asdict(config),
        'baseline': {
            'out_dist_accuracy': baseline_out['accuracy'],
            'in_dist_accuracy': baseline_in['accuracy'],
        },
        'trained': {
            'out_dist_accuracy': final_out_eval['accuracy'],
            'in_dist_accuracy': final_in_eval['accuracy'],
            'best_out_dist_accuracy': best_val_accuracy,
            'error_type_breakdown': final_out_eval.get('error_type_breakdown', {}),
        },
        'improvement': {
            'out_dist_accuracy_gain': final_out_eval['accuracy'] - baseline_out['accuracy'],
            'in_dist_accuracy_gain': final_in_eval['accuracy'] - baseline_in['accuracy'],
        },
        'training_time_minutes': elapsed / 60,
        'data': {
            'train_samples': len(train_dataset),
            'in_dist_val_samples': len(in_dist_val_dataset),
            'out_dist_val_samples': len(out_dist_val_dataset),
            'training_file': train_data_file,
            'out_dist_validation_file': out_dist_val_file,
            'use_cot': config.use_cot,
            'training_label_type': 'CARA CoT' if config.use_cot else 'CARA labels',
            'in_dist_validation_label_type': 'CARA CoT' if config.use_cot else 'CARA labels',
            'out_dist_validation_label_type': 'CARA CoT (out-of-distribution)' if config.use_cot else 'Cooperate labels (out-of-distribution)',
        }
    }

    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  Training complete in {elapsed/60:.1f} minutes")
    print(f"  Final out-dist accuracy: {final_out_eval['accuracy']:.4f}")
    print(f"  Best out-dist accuracy: {best_val_accuracy:.4f}")

    # Clean up model to free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reward model evaluation (post-training)
    if config.run_reward_eval:
        checkpoint_dir = os.path.join(output_dir, 'best_checkpoint')
        if os.path.exists(checkpoint_dir):
            reward_eval = run_reward_model_evaluation(config, checkpoint_dir, output_dir, device)
            if reward_eval:
                results['reward_eval'] = reward_eval
                # Re-save results with reward eval included
                with open(os.path.join(output_dir, 'results.json'), 'w') as f:
                    json.dump(results, f, indent=2)

    return results


# ============================================================
# SWEEP RUNNER
# ============================================================
def run_sweep(configs: List[SweepConfig], output_base: str):
    """Run sweep over all configurations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(output_base, f"sweep_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting Hyperparameter Sweep")
    print(f"{'='*60}")
    print(f"Output directory: {sweep_dir}")
    print(f"Number of configurations: {len(configs)}")
    print(f"Configurations:")
    for i, cfg in enumerate(configs):
        print(f"  {i+1}. {cfg.name}: lr={cfg.learning_rate}, lora_r={cfg.lora_r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    all_results = []
    status_records = []
    status_path = os.path.join(sweep_dir, 'run_status.json')
    sweep_start = time.time()

    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{len(configs)}: {config.name}")
        print(f"{'='*60}")

        run_dir = os.path.join(sweep_dir, f"run_{i+1}_{config.name}")
        run_start = time.time()
        record = {'run_index': i + 1, 'name': config.name, 'status': 'pending'}
        try:
            result = train_single_run(config, run_dir, device)
            result['run_index'] = i + 1
            all_results.append(result)
            record.update({
                'status': 'succeeded',
                'training_time_minutes': result.get('training_time_minutes'),
                'best_out_dist_accuracy': result.get('trained', {}).get('best_out_dist_accuracy'),
            })
        except Exception as e:
            print(f"ERROR: Run {config.name} failed: {e}")
            record.update({
                'status': 'failed',
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
            })
            all_results.append({
                'config': asdict(config),
                'error': str(e),
                'run_index': i + 1,
                'trained': {'out_dist_accuracy': 0.0, 'in_dist_accuracy': 0.0, 'best_out_dist_accuracy': 0.0, 'error_type_breakdown': {}},
                'baseline': {'out_dist_accuracy': 0.0, 'in_dist_accuracy': 0.0},
                'improvement': {'out_dist_accuracy_gain': 0.0, 'in_dist_accuracy_gain': 0.0},
                'training_time_minutes': 0.0,
                'data': {
                    'train_samples': 0, 'in_dist_val_samples': 0, 'out_dist_val_samples': 0,
                    'training_file': COT_TRAIN_DATA_FILE if config.use_cot else TRAIN_DATA_FILE,
                    'out_dist_validation_file': COT_VAL_DATA_FILE if config.use_cot else LABEL_VAL_DATA_FILE,
                    'use_cot': config.use_cot,
                    'training_label_type': 'CARA CoT' if config.use_cot else 'CARA labels',
                    'in_dist_validation_label_type': 'CARA CoT' if config.use_cot else 'CARA labels',
                    'out_dist_validation_label_type': 'CARA CoT (out-of-distribution)' if config.use_cot else 'Cooperate labels (out-of-distribution)',
                },
            })
        finally:
            record['wall_seconds'] = round(time.time() - run_start, 1)
            status_records.append(record)
            with open(status_path, 'w') as f:
                json.dump({
                    'sweep_dir': sweep_dir,
                    'updated_at': datetime.now().isoformat(timespec='seconds'),
                    'total_planned': len(configs),
                    'completed': len(status_records),
                    'succeeded': sum(1 for r in status_records if r['status'] == 'succeeded'),
                    'failed': sum(1 for r in status_records if r['status'] == 'failed'),
                    'runs': status_records,
                }, f, indent=2)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    sweep_elapsed = time.time() - sweep_start

    # Generate summary
    generate_sweep_summary(all_results, sweep_dir, sweep_elapsed)
    generate_comparison_plot(all_results, sweep_dir)

    print(f"\n{'='*60}")
    print(f"Sweep Complete!")
    print(f"{'='*60}")
    print(f"Total time: {sweep_elapsed/60:.1f} minutes")
    print(f"Results saved to: {sweep_dir}")


def generate_sweep_summary(results: List[dict], sweep_dir: str, total_time: float):
    """Generate sweep summary JSON."""
    if not results:
        print("No results to summarize.")
        return

    # Sort by out-dist accuracy
    sorted_results = sorted(
        results,
        key=lambda x: x['trained']['out_dist_accuracy'],
        reverse=True
    )

    runs_ranked = []
    for rank, r in enumerate(sorted_results, 1):
        entry = {
            'rank': rank,
            'name': r['config']['name'],
            'learning_rate': r['config']['learning_rate'],
            'lora_r': r['config']['lora_r'],
            'out_dist_accuracy': r['trained']['out_dist_accuracy'],
            'in_dist_accuracy': r['trained']['in_dist_accuracy'],
            'best_out_dist_accuracy': r['trained']['best_out_dist_accuracy'],
            'training_time_minutes': r['training_time_minutes'],
        }
        # reward_eval is a dict keyed by spec name; flatten primary metrics per spec.
        reward_eval = r.get('reward_eval') or {}
        for spec_key, spec_summary in reward_eval.items():
            if not isinstance(spec_summary, dict):
                continue
            entry[f'pairwise_accuracy__{spec_key}'] = spec_summary.get('pairwise_accuracy')
            entry[f'mean_score_margin__{spec_key}'] = spec_summary.get('mean_score_margin')
            entry[f'tie_rate__{spec_key}'] = spec_summary.get('tie_rate')
        runs_ranked.append(entry)

    # Extract data source info from first result
    first_data = results[0].get('data', {})
    summary = {
        'sweep_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'total_runs': len(results),
        'total_time_minutes': total_time / 60,
        'data_sources': {
            'training_file': first_data.get('training_file', 'unknown'),
            'out_dist_validation_file': first_data.get('out_dist_validation_file', 'unknown'),
            'training_label_type': first_data.get('training_label_type', 'unknown'),
            'out_dist_validation_label_type': first_data.get('out_dist_validation_label_type', 'unknown'),
        },
        'best_config': {
            'name': runs_ranked[0]['name'],
            'out_dist_accuracy': runs_ranked[0]['out_dist_accuracy'],
        },
        'runs_ranked': runs_ranked,
    }

    with open(os.path.join(sweep_dir, 'sweep_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary — one Pairwise-Acc column per reward-eval spec actually run.
    spec_keys_present = [k for k, _, _, _ in REWARD_EVAL_SPECS
                         if any(f'pairwise_accuracy__{k}' in r for r in runs_ranked)]
    spec_labels = {k: lbl for k, lbl, _, _ in REWARD_EVAL_SPECS}

    print(f"\n{'='*88}")
    print("SWEEP SUMMARY")
    print(f"{'='*88}")
    header = f"{'Rank':<6} {'Name':<15} {'Out-Dist Acc':>12} {'In-Dist Acc':>12}"
    for k in spec_keys_present:
        header += f" {'PW Acc ' + spec_labels[k]:>14}"
    header += f" {'Time (min)':>10}"
    print(header)
    print("-" * 88)
    for r in runs_ranked:
        line = f"{r['rank']:<6} {r['name']:<15} {r['out_dist_accuracy']:>11.4f} {r['in_dist_accuracy']:>12.4f}"
        for k in spec_keys_present:
            pw = r.get(f'pairwise_accuracy__{k}')
            line += f" {pw:>13.3f}" if pw is not None else f" {'N/A':>13}"
        line += f" {r['training_time_minutes']:>10.1f}"
        print(line)
    print(f"\nBest: {runs_ranked[0]['name']} with {runs_ranked[0]['out_dist_accuracy']:.4f} accuracy")

    data_info = summary.get('data_sources', {})
    print(f"\nData sources:")
    print(f"  Training: {data_info.get('training_label_type', 'unknown')} ({os.path.basename(data_info.get('training_file', 'unknown'))})")
    print(f"  Out-dist validation: {data_info.get('out_dist_validation_label_type', 'unknown')} ({os.path.basename(data_info.get('out_dist_validation_file', 'unknown'))})")


def generate_comparison_plot(results: List[dict], sweep_dir: str):
    """Generate comparison bar chart with optional reward-eval pairwise-accuracy subplot.

    Second subplot shows one bar group per REWARD_EVAL_SPECS spec (e.g. clean_400 vs
    rebels_500), so you can see both the externally-comparable score and the noisy-row
    score side-by-side.
    """
    names = [r['config']['name'] for r in results]
    out_accs = [r['trained']['out_dist_accuracy'] for r in results]
    in_accs = [r['trained']['in_dist_accuracy'] for r in results]
    baseline_out = results[0]['baseline']['out_dist_accuracy']

    # Per-spec pairwise accuracy: dict of spec_key -> list aligned with `results`.
    pw_by_spec = {}
    for spec_key, spec_label, _, _ in REWARD_EVAL_SPECS:
        vals = [(r.get('reward_eval') or {}).get(spec_key, {}).get('pairwise_accuracy')
                if isinstance((r.get('reward_eval') or {}).get(spec_key), dict) else None
                for r in results]
        if any(v is not None for v in vals):
            pw_by_spec[spec_key] = (spec_label, vals)
    has_pw = bool(pw_by_spec)

    if has_pw:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 2])
    else:
        fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, out_accs, width, label='Out-of-Dist', color='steelblue')
    bars2 = ax.bar(x + width/2, in_accs, width, label='In-Dist', color='coral')

    # Baseline line
    ax.axhline(y=baseline_out, color='gray', linestyle='--', label=f'Baseline ({baseline_out:.3f})')

    # Highlight best
    best_idx = np.argmax(out_accs)
    bars1[best_idx].set_color('darkgreen')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Hyperparameter Sweep Results', fontsize=14, fontweight='bold')

    # Add subtitle with label type info
    first_data = results[0].get('data', {})
    train_label = first_data.get('training_label_type', 'CARA')
    val_label = first_data.get('out_dist_validation_label_type', 'cooperate')
    ax.text(0.5, 1.02, f'Train: {train_label} | Out-dist Val: {val_label}',
            transform=ax.transAxes, ha='center', fontsize=9, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, max(max(out_accs), max(in_accs)) * 1.1)

    # Add value labels
    for bar, val in zip(bars1, out_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Pairwise-accuracy subplot — grouped bars, one group per reward-eval spec.
    if has_pw:
        n_specs = len(pw_by_spec)
        group_width = 0.8
        bar_w = group_width / n_specs
        spec_palette = ['seagreen', 'steelblue', 'mediumorchid', 'coral']

        for i, (spec_key, (spec_label, vals)) in enumerate(pw_by_spec.items()):
            offsets = x - group_width/2 + bar_w/2 + i*bar_w
            plot_vals = [v if v is not None else 0 for v in vals]
            color = spec_palette[i % len(spec_palette)]
            colors = [color if v is not None else 'lightgray' for v in vals]
            bars = ax2.bar(offsets, plot_vals, bar_w, color=colors, label=f'PW Acc — {spec_label}')

            valid = [(j, v) for j, v in enumerate(vals) if v is not None]
            if valid:
                best_j = max(valid, key=lambda t: t[1])[0]
                bars[best_j].set_edgecolor('black')
                bars[best_j].set_linewidth(1.5)

            for bar, val, raw in zip(bars, plot_vals, vals):
                if raw is not None:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                             f'{val:.2f}', ha='center', va='bottom', fontsize=7)
                else:
                    ax2.text(bar.get_x() + bar.get_width()/2, 0.05,
                             'N/A', ha='center', va='bottom', fontsize=7, color='gray')

        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Pairwise Accuracy')
        ax2.set_title('Reward Model Eval: Pairwise Accuracy (chosen > rejected)', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylim(0, 1.1)
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random (0.5)')
        ax2.legend(fontsize=8, loc='upper right')

    # Add data file footer
    train_file = os.path.basename(first_data.get('training_file', ''))
    val_file = os.path.basename(first_data.get('out_dist_validation_file', ''))
    if train_file or val_file:
        fig.text(0.5, 0.01, f'Data: {train_file} / {val_file}',
                 ha='center', fontsize=7, color='gray', style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(sweep_dir, 'sweep_comparison.png'), dpi=150)
    plt.close()
    print(f"\nComparison plot saved to: {sweep_dir}/sweep_comparison.png")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Sweep for Reward Model Training')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help='Base output directory for sweep results')
    parser.add_argument('--configs', type=str, default=None,
                        help='Comma-separated run indices (1-indexed), e.g., "1,3,5"')
    parser.add_argument('--list-configs', action='store_true',
                        help='List available configurations and exit')
    args = parser.parse_args()

    if args.list_configs:
        print("Available configurations:")
        for i, cfg in enumerate(SWEEP_CONFIGS, 1):
            print(f"  {i}. {cfg.name}: lr={cfg.learning_rate}, lora_r={cfg.lora_r}, lora_alpha={cfg.lora_alpha}")
        return

    configs = SWEEP_CONFIGS
    if args.configs:
        indices = [int(x.strip()) - 1 for x in args.configs.split(",")]
        for i in indices:
            if i < 0 or i >= len(SWEEP_CONFIGS):
                print(f"Error: Config index {i+1} out of range (1-{len(SWEEP_CONFIGS)})")
                sys.exit(1)
        configs = [SWEEP_CONFIGS[i] for i in indices]
        print(f"Running subset of configs: {[c.name for c in configs]}")

    run_sweep(configs, args.output_dir)


if __name__ == "__main__":
    main()
