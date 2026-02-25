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
import json
import time
import argparse
import subprocess
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
    learning_rate: float = 2e-4
    reward_head_lr_multiplier: float = 2.5  # Reward head LR = learning_rate * this
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    # Fixed params
    model_name: str = "Qwen/Qwen3-8B"
    batch_size: int = 2
    gradient_accumulation_steps: int = 32
    num_epochs: int = 10
    weight_decay: float = 0.01
    max_length: int = 256
    in_dist_val_split: float = 0.10
    random_seed: int = 42
    # CoT training options
    use_cot: bool = True
    cot_max_length: int = 1024  # Reduced for memory efficiency (actual usage ~515 tokens)
    # Learning rate scheduler
    use_lr_scheduler: bool = True
    warmup_ratio: float = 0.05  # 5% of total steps for warmup
    # Gradient clipping
    use_gradient_clipping: bool = True
    max_grad_norm: float = 1.0  # Standard default for LLM training
    # Reference model for KL regularization
    use_reference_model: bool = True
    reference_kl_beta: float = 0.1  # KL penalty weight (0 = monitoring only)
    # Model initialization
    reward_head_init_std: float = 0.01  # Std dev for reward head weight init
    # Evaluation
    eval_batch_size: int = 4  # Batch size for evaluation (no gradients, can be larger)
    # Generative evaluation (post-training)
    run_generative_eval: bool = True
    generative_eval_num_situations: int = 50
    generative_eval_temperature: float = 0.0  # deterministic for reproducibility


# Default sweep configurations
# Note: use_cot defaults to True, so label-only configs explicitly set use_cot=False
SWEEP_CONFIGS = [
    # Label-only configurations (for comparison)
    SweepConfig(name="label_lr_1e-4", learning_rate=1e-4, use_cot=False),
    SweepConfig(name="label_lr_2e-4", learning_rate=2e-4, use_cot=False),  # label-only baseline
    # CoT (Chain-of-Thought) configurations - default
    SweepConfig(name="cot_lr_5e-5", learning_rate=5e-5),
    SweepConfig(name="cot_lr_1e-4", learning_rate=1e-4),
    SweepConfig(name="cot_lr_2e-4", learning_rate=2e-4),  # CoT baseline
    SweepConfig(name="cot_lr_5e-4", learning_rate=5e-4),
    SweepConfig(name="cot_lora_r4", lora_r=4, lora_alpha=8),
    SweepConfig(name="cot_lora_r16", lora_r=16, lora_alpha=32),
    # Reward head LR multiplier ablation
    SweepConfig(name="head_mult_1.0x", reward_head_lr_multiplier=1.0),
    SweepConfig(name="head_mult_2.0x", reward_head_lr_multiplier=2.0),
    SweepConfig(name="head_mult_5.0x", reward_head_lr_multiplier=5.0),
]

# Data file paths
# Note: The old label-only training file no longer exists; use_cot=False configs will fail
# The new training file has 1000 situations (500 old + 500 new) with more balanced rejected types
COT_TRAIN_DATA_FILE = "data/2026_01_29_new_full_training_set_with_CoTs_Sonnet_4_5.csv"
VAL_DATA_FILE = "data/2026_02_11_val_set_CoTs_from_Sonnet.csv"
# Legacy path (no longer exists, kept for reference)
TRAIN_DATA_FILE = COT_TRAIN_DATA_FILE  # Fallback to CoT file
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
        lora_r: int = 8,
        lora_alpha: int = 16,
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
            lora_target_modules = ["q_proj", "v_proj"]

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


def evaluate_model(model, dataset, device, batch_size=4):
    """Evaluate model on pairwise accuracy."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    total_loss = 0.0
    preferred_scores = []
    rejected_scores = []
    correct_by_type = {'too_risky': 0, 'too_risk_averse': 0, 'other': 0}
    total_by_type = {'too_risky': 0, 'too_risk_averse': 0, 'other': 0}

    with torch.no_grad():
        for batch in dataloader:
            try:
                preferred_rewards = model(
                    input_ids=batch['preferred_input_ids'].to(device),
                    attention_mask=batch['preferred_attention_mask'].to(device)
                )
                rejected_rewards = model(
                    input_ids=batch['rejected_input_ids'].to(device),
                    attention_mask=batch['rejected_attention_mask'].to(device)
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  WARNING: GPU OOM during evaluation. Clearing cache and skipping batch.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

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

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0

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
# Path to the eval repo submodule
EVAL_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "risk-averse-ai-eval")
EVAL_SCRIPT = os.path.join(EVAL_REPO_DIR, "evaluate.py")
EVAL_VAL_CSV = os.path.join(EVAL_REPO_DIR, "data", "2026_01_29_new_val_set_probabilities_add_to_100.csv")


def run_generative_evaluation(config: SweepConfig, checkpoint_dir: str, output_dir: str, device: torch.device) -> Optional[dict]:
    """Run generative evaluation using risk-averse-ai-eval on saved checkpoint.

    Invokes the eval repo's evaluate.py via subprocess after training is complete
    and the training model has been freed from GPU memory.

    Returns a summary dict with key metrics, or None if evaluation could not run.
    """
    # Check that the eval repo submodule exists
    if not os.path.isfile(EVAL_SCRIPT):
        print(f"  WARNING: Generative eval skipped - eval script not found at {EVAL_SCRIPT}")
        print(f"  To enable: git submodule update --init")
        return None

    if not os.path.isfile(EVAL_VAL_CSV):
        print(f"  WARNING: Generative eval skipped - val CSV not found at {EVAL_VAL_CSV}")
        return None

    if not os.path.isdir(checkpoint_dir):
        print(f"  WARNING: Generative eval skipped - checkpoint not found at {checkpoint_dir}")
        return None

    output_json = os.path.join(output_dir, "generative_eval_results.json")

    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--model_path", checkpoint_dir,
        "--base_model", config.model_name,
        "--val_csv", EVAL_VAL_CSV,
        "--num_situations", str(config.generative_eval_num_situations),
        "--temperature", str(config.generative_eval_temperature),
        "--output", output_json,
        "--no_save_responses",
        "--disable_thinking",
    ]

    print(f"\n  Running generative evaluation ({config.generative_eval_num_situations} situations, temp={config.generative_eval_temperature})...")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )

        if result.returncode != 0:
            print(f"  WARNING: Generative eval failed (exit code {result.returncode})")
            if result.stderr:
                # Print last 20 lines of stderr for debugging
                stderr_lines = result.stderr.strip().split('\n')
                for line in stderr_lines[-20:]:
                    print(f"    {line}")
            return None

        # Parse output JSON
        if not os.path.isfile(output_json):
            print(f"  WARNING: Generative eval produced no output file")
            return None

        with open(output_json, 'r') as f:
            eval_results = json.load(f)

        metrics = eval_results.get('metrics', {})
        summary = {
            'cara_rate': metrics.get('best_cara_rate', None),
            'parse_rate': metrics.get('parse_rate', None),
            'cooperate_rate': metrics.get('cooperate_rate', None),
            'rebel_rate': metrics.get('rebel_rate', None),
            'steal_rate': metrics.get('steal_rate', None),
            'best_linear_rate': metrics.get('best_linear_rate', None),
            'num_situations': eval_results.get('num_total', config.generative_eval_num_situations),
            'num_valid': eval_results.get('num_valid', None),
            'temperature': config.generative_eval_temperature,
            'eval_dataset': os.path.basename(EVAL_VAL_CSV),
        }

        # Print summary
        cara = summary['cara_rate']
        parse = summary['parse_rate']
        print(f"  Generative eval complete:")
        print(f"    CARA rate: {cara:.3f}" if cara is not None else "    CARA rate: N/A")
        print(f"    Parse rate: {parse:.3f}" if parse is not None else "    Parse rate: N/A")
        print(f"    Cooperate: {summary['cooperate_rate']}" if summary['cooperate_rate'] is not None else "")
        print(f"    Rebel: {summary['rebel_rate']}" if summary['rebel_rate'] is not None else "")
        print(f"    Steal: {summary['steal_rate']}" if summary['steal_rate'] is not None else "")

        return summary

    except subprocess.TimeoutExpired:
        print(f"  WARNING: Generative eval timed out after 30 minutes")
        return None
    except Exception as e:
        print(f"  WARNING: Generative eval error: {e}")
        return None


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

    # Validate config
    if not config.use_cot:
        raise ValueError(
            f"Label-only training is no longer supported. "
            f"Config '{config.name}' has use_cot=False but the label-only training file no longer exists. "
            f"Set use_cot=True or remove this config."
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine which data file and max_length to use based on CoT setting
    if config.use_cot:
        train_data_file = COT_TRAIN_DATA_FILE
        train_max_length = config.cot_max_length
        print(f"  Using CoT training data: {train_data_file}")
        print(f"  CoT max_length: {train_max_length}")
    else:
        train_data_file = TRAIN_DATA_FILE
        train_max_length = config.max_length

    # Load data with train/in-dist-val split
    train_ids, in_dist_val_ids = get_train_val_situation_split(
        train_data_file,
        val_fraction=config.in_dist_val_split,
        random_seed=config.random_seed
    )

    # Load datasets based on CoT setting
    if config.use_cot:
        train_loader = CoTTrainingDataLoader(
            train_data_file, random_seed=config.random_seed, situation_ids=train_ids
        )
    else:
        train_loader = TrainingDataLoader(
            train_data_file, epoch=0, random_seed=config.random_seed, situation_ids=train_ids
        )
    train_df = train_loader.load_and_process_data()
    if train_df.empty:
        raise ValueError(f"Training data is empty after processing '{train_data_file}'. Check data file and filters.")

    # In-dist validation uses same data source as training
    if config.use_cot:
        in_dist_val_loader = CoTTrainingDataLoader(
            train_data_file, random_seed=config.random_seed, situation_ids=in_dist_val_ids
        )
    else:
        in_dist_val_loader = InDistributionValidationDataLoader(
            train_data_file, random_seed=config.random_seed, situation_ids=in_dist_val_ids
        )
    in_dist_val_df = in_dist_val_loader.load_and_process_data()
    if in_dist_val_df.empty:
        raise ValueError(f"In-distribution validation data is empty after processing. Check data file and situation ID split.")

    # Out-dist validation uses CoT format (same loader as training, different situations)
    out_dist_val_loader = CoTTrainingDataLoader(VAL_DATA_FILE, random_seed=config.random_seed)
    out_dist_val_df = out_dist_val_loader.load_and_process_data()
    if out_dist_val_df.empty:
        raise ValueError(f"Out-of-distribution validation data is empty after processing '{VAL_DATA_FILE}'. Check data file.")

    # Create datasets with appropriate max_length (all use CoT length since all are CoT format)
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
    print("\nBaseline evaluation...")
    eval_bs = config.eval_batch_size
    baseline_out = evaluate_model(model, out_dist_val_dataset, device, batch_size=eval_bs)
    baseline_in = evaluate_model(model, in_dist_val_dataset, device, batch_size=eval_bs)
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

    # Training loop
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Recreate training dataset with epoch-specific randomization
        # Note: CoT data is pre-generated so no per-epoch randomization needed
        if epoch > 0 and not config.use_cot:
            train_loader = TrainingDataLoader(
                train_data_file, epoch=epoch, random_seed=config.random_seed, situation_ids=train_ids
            )
            train_df = train_loader.load_and_process_data()
            train_dataset = PairwiseRewardDataset(train_df, tokenizer, max_length=train_max_length)

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
                    print(f"  WARNING: GPU OOM at epoch {epoch+1}, step {step+1}. Clearing cache and skipping batch.")
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    oom_encountered = True
                    continue
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
            'out_dist_validation_file': VAL_DATA_FILE,
            'use_cot': config.use_cot,
            'training_label_type': 'CARA CoT' if config.use_cot else 'CARA (risk-aversion)',
            'in_dist_validation_label_type': 'CARA CoT' if config.use_cot else 'CARA (risk-aversion)',
            'out_dist_validation_label_type': 'CARA CoT (out-of-distribution)',
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

    # Generative evaluation (post-training)
    if config.run_generative_eval:
        checkpoint_dir = os.path.join(output_dir, 'best_checkpoint')
        if os.path.exists(checkpoint_dir):
            gen_eval = run_generative_evaluation(config, checkpoint_dir, output_dir, device)
            if gen_eval:
                results['generative_eval'] = gen_eval
                # Re-save results with generative eval included
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
    sweep_start = time.time()

    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{len(configs)}: {config.name}")
        print(f"{'='*60}")

        run_dir = os.path.join(sweep_dir, f"run_{i+1}_{config.name}")
        try:
            result = train_single_run(config, run_dir, device)
            result['run_index'] = i + 1
            all_results.append(result)
        except Exception as e:
            print(f"ERROR: Run {config.name} failed: {e}")
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
                    'out_dist_validation_file': VAL_DATA_FILE,
                    'use_cot': config.use_cot,
                    'training_label_type': 'CARA CoT' if config.use_cot else 'CARA (risk-aversion)',
                    'in_dist_validation_label_type': 'CARA CoT' if config.use_cot else 'CARA (risk-aversion)',
                    'out_dist_validation_label_type': 'CARA CoT (out-of-distribution)',
                },
            })

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
        gen_eval = r.get('generative_eval', {})
        if gen_eval:
            entry['cara_rate'] = gen_eval.get('cara_rate')
            entry['parse_rate'] = gen_eval.get('parse_rate')
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

    # Print summary
    has_cara = any('cara_rate' in r for r in runs_ranked)
    print(f"\n{'='*72}")
    print("SWEEP SUMMARY")
    print(f"{'='*72}")
    header = f"{'Rank':<6} {'Name':<15} {'Out-Dist Acc':>12} {'In-Dist Acc':>12}"
    if has_cara:
        header += f" {'CARA Rate':>10}"
    header += f" {'Time (min)':>10}"
    print(header)
    print("-" * 72)
    for r in runs_ranked:
        line = f"{r['rank']:<6} {r['name']:<15} {r['out_dist_accuracy']:>11.4f} {r['in_dist_accuracy']:>12.4f}"
        if has_cara:
            cara = r.get('cara_rate')
            line += f" {cara:>9.3f}" if cara is not None else f" {'N/A':>9}"
        line += f" {r['training_time_minutes']:>10.1f}"
        print(line)
    print(f"\nBest: {runs_ranked[0]['name']} with {runs_ranked[0]['out_dist_accuracy']:.4f} accuracy")

    data_info = summary.get('data_sources', {})
    print(f"\nData sources:")
    print(f"  Training: {data_info.get('training_label_type', 'unknown')} ({os.path.basename(data_info.get('training_file', 'unknown'))})")
    print(f"  Out-dist validation: {data_info.get('out_dist_validation_label_type', 'unknown')} ({os.path.basename(data_info.get('out_dist_validation_file', 'unknown'))})")


def generate_comparison_plot(results: List[dict], sweep_dir: str):
    """Generate comparison bar chart with optional CARA rate subplot."""
    names = [r['config']['name'] for r in results]
    out_accs = [r['trained']['out_dist_accuracy'] for r in results]
    in_accs = [r['trained']['in_dist_accuracy'] for r in results]
    baseline_out = results[0]['baseline']['out_dist_accuracy']

    # Check if any results have generative eval data
    cara_rates = [r.get('generative_eval', {}).get('cara_rate') for r in results]
    has_cara = any(c is not None for c in cara_rates)

    if has_cara:
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

    # CARA rate subplot
    if has_cara:
        # Replace None with 0 for plotting, track which are valid
        cara_vals = [c if c is not None else 0 for c in cara_rates]
        cara_colors = ['seagreen' if c is not None else 'lightgray' for c in cara_rates]

        bars3 = ax2.bar(x, cara_vals, width=0.5, color=cara_colors)

        # Highlight best CARA rate
        valid_cara = [(i, c) for i, c in enumerate(cara_rates) if c is not None]
        if valid_cara:
            best_cara_idx = max(valid_cara, key=lambda t: t[1])[0]
            bars3[best_cara_idx].set_color('darkgreen')

        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('CARA Rate')
        ax2.set_title('Generative Evaluation: CARA Rate (% choosing risk-averse option)', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylim(0, 1.1)
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random (0.5)')
        ax2.legend(fontsize=8)

        # Add value labels on CARA bars
        for bar, val, raw in zip(bars3, cara_vals, cara_rates):
            if raw is not None:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                ax2.text(bar.get_x() + bar.get_width()/2, 0.05,
                         'N/A', ha='center', va='bottom', fontsize=8, color='gray')

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
