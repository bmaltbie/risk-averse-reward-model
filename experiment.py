#!/usr/bin/env python3
"""
Risk-Averse Reward Model Training Script

Standalone Python script for training risk-averse reward models on cloud GPUs.
Converted from colab_notebook.ipynb for SSH/CLI execution.

Usage:
    python experiment.py --model Qwen/Qwen3-8B --data data/11_7_low_stakes_training_set.csv
"""

# Set matplotlib backend FIRST (before any other imports)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environment

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
import json
from typing import Dict, Tuple
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


def setup_logging(output_dir: str = "logs") -> logging.Logger:
    """Configure logging to both file and console"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"experiment_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging to: {log_file}")
    return logger


# ============================================================================
# Classes
# ============================================================================

class RiskAversionDataLoader:
    """Load and process data from CSV file for risk aversion training"""

    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path

    def load_and_process_data(self) -> pd.DataFrame:
        """Load CSV data and process it for training"""
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(
                f"Required data file '{self.csv_file_path}' not found."
            )

        df = pd.read_csv(self.csv_file_path)
        logging.info(f"Loaded {len(df)} rows from {self.csv_file_path}")

        # Check required columns
        required_columns = ['situation_id', 'prompt_text', 'option_index', 'is_best_cara', 'is_best_linear']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Available: {list(df.columns)}"
            )

        # Group by situation_id
        situations = []
        situations_skipped = 0

        for situation_id, group in df.groupby('situation_id'):
            # Find risk-averse option (CARA best)
            cara_rows = group[group['is_best_cara'] == True]
            if len(cara_rows) == 0:
                situations_skipped += 1
                continue
            cara_option = cara_rows.iloc[0]

            # Find risk-neutral option (LINEAR best)
            linear_rows = group[group['is_best_linear'] == True]
            if len(linear_rows) == 0:
                situations_skipped += 1
                continue
            linear_option = linear_rows.iloc[0]

            prompt_text = group.iloc[0]['prompt_text']

            # Convert 0-indexed to 1-indexed
            correct_label = str(cara_option['option_index'] + 1)
            incorrect_label = str(linear_option['option_index'] + 1)

            situations.append({
                'situation_id': situation_id,
                'prompt_text': prompt_text,
                'correct_label': correct_label,
                'incorrect_label': incorrect_label,
                'num_options': len(group)
            })

        if situations_skipped > 0:
            logging.warning(f"Skipped {situations_skipped} situations missing CARA or LINEAR best")

        result_df = pd.DataFrame(situations)
        logging.info(f"Processed into {len(result_df)} unique situations")

        if len(result_df) > 0:
            sample = result_df.iloc[0]
            logging.info(f"Sample - Risk-averse: Option {sample['correct_label']}, "
                        f"Risk-neutral: Option {sample['incorrect_label']}")

        return result_df


class SingleInputDataset(Dataset):
    """Dataset for pure single-input classification training"""

    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = 128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Expand dataset: 2 examples per situation
        self.examples = []
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            # Risk-averse example
            self.examples.append({
                'situation_idx': idx,
                'is_risk_averse': True,
                'situation_id': row['situation_id']
            })
            # Risk-neutral example
            self.examples.append({
                'situation_idx': idx,
                'is_risk_averse': False,
                'situation_id': row['situation_id']
            })

        logging.info(f"SingleInputDataset: {len(self.examples)} examples from {len(self.data)} situations")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example_info = self.examples[idx]
        row = self.data.iloc[example_info['situation_idx']]

        is_risk_averse = example_info['is_risk_averse']
        option_text = row['correct_label'] if is_risk_averse else row['incorrect_label']
        input_text = f"{row['prompt_text']}\n\nChosen option: {option_text}"
        label = 1.0 if is_risk_averse else 0.0

        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label,
            'situation_id': row['situation_id']
        }


class RiskAverseRewardModel(nn.Module):
    """Reward model for scoring risk-averse behavior"""

    def __init__(self, model_name: str):
        super().__init__()
        load_kwargs = {
            "num_labels": 1,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "torch_dtype": "auto",
        }

        logging.info(f"Loading model: {model_name}")
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **load_kwargs
        )

        self.backbone.gradient_checkpointing_enable()
        logging.info("Gradient checkpointing enabled")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Single-input forward pass"""
        device = next(self.backbone.parameters()).device

        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits.squeeze(-1)

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

            if self.training and torch.rand(1).item() < 0.01:
                pred_probs = torch.sigmoid(logits).mean().item()
                logging.info(f"[TRAIN] Avg logit: {logits.mean().item():.3f}, "
                           f"Avg sigmoid: {pred_probs:.3f}, Target: {labels.mean().item():.3f}")

            return {"loss": loss, "logits": logits}

        return {"logits": logits}


# ============================================================================
# Functions
# ============================================================================

def evaluate_model(model, tokenizer, test_df: pd.DataFrame, max_length: int = 128,
                  return_detailed: bool = False):
    """Evaluate the trained model"""
    logging.info(f"Evaluating model on {len(test_df)} test situations...")

    model.eval()
    correct_predictions = 0
    total_predictions = 0
    risk_averse_wins = 0

    results = {
        'risk_averse_scores': [],
        'risk_neutral_scores': [],
        'predictions': [],
        'expected': [],
        'situation_ids': [],
    }

    device = next(model.parameters()).device

    with torch.no_grad():
        for idx, row in test_df.iterrows():
            # Risk-averse option
            risk_averse_text = f"{row['prompt_text']}\n\nChosen option: {row['correct_label']}"
            risk_averse_encoding = tokenizer(
                risk_averse_text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            risk_averse_encoding = {k: v.to(device) for k, v in risk_averse_encoding.items()}
            risk_averse_output = model(**risk_averse_encoding)
            risk_averse_score = risk_averse_output["logits"].item()

            # Risk-neutral option
            risk_neutral_text = f"{row['prompt_text']}\n\nChosen option: {row['incorrect_label']}"
            risk_neutral_encoding = tokenizer(
                risk_neutral_text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            risk_neutral_encoding = {k: v.to(device) for k, v in risk_neutral_encoding.items()}
            risk_neutral_output = model(**risk_neutral_encoding)
            risk_neutral_score = risk_neutral_output["logits"].item()

            # Binary classification accuracy
            risk_averse_pred = torch.sigmoid(torch.tensor(risk_averse_score)).item()
            risk_neutral_pred = torch.sigmoid(torch.tensor(risk_neutral_score)).item()

            if risk_averse_pred > 0.5:
                correct_predictions += 1
            if risk_neutral_pred <= 0.5:
                correct_predictions += 1
            total_predictions += 2

            if risk_averse_score > risk_neutral_score:
                risk_averse_wins += 1

            results['risk_averse_scores'].append(risk_averse_score)
            results['risk_neutral_scores'].append(risk_neutral_score)
            results['predictions'].extend([risk_averse_pred, risk_neutral_pred])
            results['expected'].extend([1.0, 0.0])
            results['situation_ids'].append(row['situation_id'])

            if (idx + 1) % 50 == 0:
                current_acc = correct_predictions / total_predictions
                current_pref = risk_averse_wins / (idx + 1)
                logging.info(f"  Progress: {idx + 1}/{len(test_df)} | "
                           f"Accuracy: {current_acc:.3f} | Risk-averse pref: {current_pref:.3f}")

    accuracy = correct_predictions / total_predictions
    risk_averse_preference_rate = risk_averse_wins / len(test_df)

    logging.info(f"Evaluation complete:")
    logging.info(f"  Accuracy: {accuracy:.3f}")
    logging.info(f"  Risk-averse preference rate: {risk_averse_preference_rate:.3f}")

    if return_detailed:
        results['risk_averse_preference_rate'] = risk_averse_preference_rate
        return accuracy, results
    return accuracy


def plot_training_loss(trainer, ax):
    """Plot training and validation loss"""
    log_history = trainer.state.log_history

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for log_entry in log_history:
        if 'loss' in log_entry:
            train_steps.append(log_entry['step'])
            train_losses.append(log_entry['loss'])
        if 'eval_loss' in log_entry:
            eval_steps.append(log_entry['step'])
            eval_losses.append(log_entry['eval_loss'])

    ax.plot(train_steps, train_losses, label='Training Loss', linewidth=2, marker='o', markersize=4)
    if eval_losses:
        ax.plot(eval_steps, eval_losses, label='Validation Loss', linewidth=2, marker='s', markersize=4)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_score_distribution(eval_results, ax):
    """Plot distribution of model scores"""
    risk_averse_scores = eval_results['risk_averse_scores']
    risk_neutral_scores = eval_results['risk_neutral_scores']

    bins = np.linspace(min(min(risk_averse_scores), min(risk_neutral_scores)),
                      max(max(risk_averse_scores), max(risk_neutral_scores)), 21)
    ax.hist(risk_averse_scores, bins=bins, alpha=0.7, label='Risk-Averse Options',
           color='green', density=True)
    ax.hist(risk_neutral_scores, bins=bins, alpha=0.7, label='Risk-Neutral Options',
           color='red', density=True)

    ax.set_xlabel('Model Score (logits)')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution by Option Type')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_risk_preference_comparison(eval_results, ax):
    """Plot comparison of scores"""
    risk_averse_scores = np.array(eval_results['risk_averse_scores'])
    risk_neutral_scores = np.array(eval_results['risk_neutral_scores'])

    ax.scatter(risk_neutral_scores, risk_averse_scores, alpha=0.6, s=50)

    min_val = min(risk_neutral_scores.min(), risk_averse_scores.min())
    max_val = max(risk_neutral_scores.max(), risk_averse_scores.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Equal Preference')

    ax.fill_between([min_val, max_val], [min_val, max_val], [max_val, max_val],
                    alpha=0.2, color='green', label='Risk-Averse Preferred')
    ax.fill_between([min_val, max_val], [min_val, min_val], [min_val, max_val],
                    alpha=0.2, color='red', label='Risk-Neutral Preferred')

    ax.set_xlabel('Risk-Neutral Option Score')
    ax.set_ylabel('Risk-Averse Option Score')
    ax.set_title('Risk Preference Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_performance_summary(eval_results, accuracy, ax):
    """Plot performance summary"""
    risk_averse_scores = np.array(eval_results['risk_averse_scores'])
    risk_neutral_scores = np.array(eval_results['risk_neutral_scores'])

    correctly_prefers_risk_averse = np.mean(risk_averse_scores > risk_neutral_scores)
    score_difference = np.mean(risk_averse_scores) - np.mean(risk_neutral_scores)

    metrics = ['Overall\nAccuracy', 'Risk-Averse\nPreference Rate']
    values = [accuracy, correctly_prefers_risk_averse]
    colors = ['blue', 'green']

    bars = ax.bar(metrics, values, color=colors, alpha=0.7)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Summary')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    ax.text(0.5, 0.5, f'Avg Score Difference:\n{score_difference:+.3f}',
           transform=ax.transAxes, ha='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontweight='bold', fontsize=12)


def plot_results(trainer, eval_results, accuracy, output_dir: str = "outputs"):
    """Create comprehensive plots (save only, no display)"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Risk-Averse Reward Model Training Results', fontsize=16, fontweight='bold')

    plot_training_loss(trainer, axes[0, 0])
    plot_score_distribution(eval_results, axes[0, 1])
    plot_risk_preference_comparison(eval_results, axes[1, 0])
    plot_performance_summary(eval_results, accuracy, axes[1, 1])

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"training_results_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close to free memory

    logging.info(f"Plots saved to {filename}")


def train_reward_model(dataset_df: pd.DataFrame, model_name: str, epochs: int,
                       batch_size: int, learning_rate: float, max_length: int,
                       output_dir: str = "./risk_averse_model"):
    """Train the risk-averse reward model"""
    logging.info(f"Training with {len(dataset_df)} situations")
    logging.info(f"Model: {model_name}")
    logging.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = RiskAverseRewardModel(model_name)
    model.backbone.config.pad_token_id = tokenizer.pad_token_id

    # Split data
    train_df, val_df = train_test_split(dataset_df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = SingleInputDataset(train_df, tokenizer, max_length)
    val_dataset = SingleInputDataset(val_df, tokenizer, max_length)

    logging.info(f"Training: {len(train_df)} situations → {len(train_dataset)} examples")
    logging.info(f"Validation: {len(val_df)} situations → {len(val_dataset)} examples")

    # Data collator
    def collate_fn(features):
        input_ids = torch.stack([f['input_ids'] for f in features])
        attention_mask = torch.stack([f['attention_mask'] for f in features])
        labels = torch.tensor([f['labels'] for f in features], dtype=torch.float32)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=50,
        report_to="none",
        eval_strategy="steps",
        eval_steps=250,
        save_steps=250,
        save_total_limit=3,
        load_best_model_at_end=False,
        fp16=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        prediction_loss_only=True,
        gradient_checkpointing=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    logging.info("Starting training...")
    trainer.train()

    return model, tokenizer, trainer


def run_experiment(args):
    """Run the complete experiment"""
    logging.info("=" * 80)
    logging.info("RISK-AVERSE REWARD MODEL EXPERIMENT")
    logging.info("=" * 80)
    logging.info(f"PyTorch device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.data}")

    # Load data
    logging.info("\n1. Loading data...")
    loader = RiskAversionDataLoader(args.data)
    dataset_df = loader.load_and_process_data()

    # Split into train/test
    train_df, test_df = train_test_split(dataset_df, test_size=0.2, random_state=42)
    logging.info(f"Train: {len(train_df)} situations")
    logging.info(f"Test: {len(test_df)} situations")

    # Train model
    logging.info("\n2. Training model...")
    model, tokenizer, trainer = train_reward_model(
        train_df,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        output_dir=os.path.join(args.output_dir, "model")
    )

    # Evaluate
    logging.info("\n3. Evaluating model...")
    accuracy, eval_results = evaluate_model(
        model, tokenizer, test_df,
        max_length=args.max_length,
        return_detailed=True
    )

    # Plot results
    logging.info("\n4. Creating visualizations...")
    plot_results(trainer, eval_results, accuracy, args.output_dir)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "num_training_situations": len(train_df),
        "num_test_situations": len(test_df),
        "final_accuracy": accuracy,
        "risk_averse_preference_rate": eval_results['risk_averse_preference_rate'],
        "model_name": args.model,
        "training_epochs": args.epochs,
        "dataset": args.data,
        "timestamp": datetime.now().isoformat()
    }

    results_file = os.path.join(args.output_dir, "experiment_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"\n{'=' * 80}")
    logging.info("EXPERIMENT COMPLETE")
    logging.info(f"{'=' * 80}")
    logging.info(f"Results saved to {results_file}")
    logging.info(f"Final accuracy: {accuracy:.3f}")
    logging.info(f"Risk-averse preference rate: {eval_results['risk_averse_preference_rate']:.3f}")

    risk_averse_scores = np.array(eval_results['risk_averse_scores'])
    risk_neutral_scores = np.array(eval_results['risk_neutral_scores'])
    score_difference = np.mean(risk_averse_scores) - np.mean(risk_neutral_scores)
    logging.info(f"Avg score difference: {score_difference:+.3f}")

    return model, tokenizer, results


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Train risk-averse reward model on cloud GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-8B',
        help='HuggingFace model name/path'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='data/11_7_low_stakes_training_set.csv',
        help='Path to training data CSV file'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Per-device training batch size'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=128,
        help='Maximum sequence length'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for results and plots'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Setup logging
    logger = setup_logging()

    try:
        # Run experiment
        model, tokenizer, results = run_experiment(args)
        logger.info("\n✓ Experiment completed successfully!")
        return 0

    except FileNotFoundError as e:
        logger.error(f"\n✗ Error: {e}")
        logger.error("Please check the data file path.")
        return 1

    except Exception as e:
        logger.error(f"\n✗ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
