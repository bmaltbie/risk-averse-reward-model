"""
Risk-Averse Reward Model Training Experiment

This experiment trains a reward model to be risk-averse using Qwen 1B model.
Uses data from strict_disagreements_10k_with_prompts_and_bad_formats.csv
Compatible with both local execution and Google Colab.
"""

import os
import sys
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
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
import json
from typing import List, Dict, Tuple
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Environment detection and setup
def setup_environment():
    """Setup environment for both local and Colab execution"""
    try:
        import google.colab
        IN_COLAB = True
        print("Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("Running locally")
    
    if IN_COLAB:
        # Install required packages in Colab
        os.system("pip install transformers datasets accelerate")
    
    return IN_COLAB

# Data loading for the experiment
class RiskAversionDataLoader:
    """Load and process data from CSV file for risk aversion training"""
    
    def __init__(self, csv_file_path="data/strict_disagreements_10k_with_prompts_and_bad_formats.csv"):
        self.csv_file_path = csv_file_path
        
    def load_and_process_data(self) -> pd.DataFrame:
        """Load CSV data and process it for training"""
        # Check if CSV file exists
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(
                f"Required data file '{self.csv_file_path}' not found. "
                f"Please ensure the CSV file is present in the current directory."
            )
        
        # Load the CSV file
        df = pd.read_csv(self.csv_file_path)
        print(f"Loaded {len(df)} rows from {self.csv_file_path}")
        
        # Check required columns exist
        required_columns = ['situation_id', 'prompt_text', 'correct_label', 'incorrect_label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in CSV: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Group by situation_id to get unique situations
        situations = []
        for situation_id, group in df.groupby('situation_id'):
            # Take the first row for each situation (they should have the same prompt_text, correct_label, incorrect_label)
            first_row = group.iloc[0]
            
            # Process the prompt text - replace thinking instruction with output-only instruction
            prompt_text = first_row['prompt_text']
            if 'You can think about which you\'d choose' in prompt_text:
                prompt_text = prompt_text.replace(
                    'You can think about which you\'d choose and then output your answer.',
                    'Output the label of your chosen option only.'
                )
            elif 'You can think about which you\'d choose' in prompt_text:
                prompt_text = prompt_text.replace(
                    'You can think about which you\'d choose and then output your answer.',
                    'Output the label of your chosen option only.'
                )
            
            situations.append({
                'situation_id': situation_id,
                'prompt_text': prompt_text,
                'correct_label': first_row['correct_label'],
                'incorrect_label': first_row['incorrect_label'],
                # Include bad versions if they exist
                'bad_correct_answers': first_row.get('bad_correct_answers', ''),
                'bad_incorrect_answers': first_row.get('bad_incorrect_answers', '')
            })
        
        result_df = pd.DataFrame(situations)
        print(f"Processed into {len(result_df)} unique situations")
        
        # Display sample data
        if len(result_df) > 0:
            print(f"\nSample situation:")
            print(f"Prompt: {result_df.iloc[0]['prompt_text'][:200]}...")
            print(f"Risk-averse choice: {result_df.iloc[0]['correct_label']}")
            print(f"Risk-neutral choice: {result_df.iloc[0]['incorrect_label']}")
        
        return result_df

# Dataset class for training
class RiskAversionDataset(Dataset):
    """Dataset for risk aversion reward model training"""
    
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length=128):  # Reduced from 512 to 128
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data) * 2  # Each situation generates 2 examples (correct + incorrect)
    
    def __getitem__(self, idx):
        situation_idx = idx // 2
        is_correct = (idx % 2) == 0
        
        row = self.data.iloc[situation_idx]
        
        # Create input text with chosen option
        chosen_label = row["correct_label"] if is_correct else row["incorrect_label"]
        input_text = f"{row['prompt_text']}\n\nChosen option: {chosen_label}"
        
        # Tokenize
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
            'labels': torch.tensor(1.0 if is_correct else 0.0, dtype=torch.float)
        }

# New pairwise dataset for ranking loss
class PairwiseRiskAversionDataset(Dataset):
    """Dataset that provides pairs of risk-averse vs risk-neutral choices for ranking loss"""
    
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)  # One pair per situation
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Create input texts for both choices
        risk_averse_text = f"{row['prompt_text']}\n\nChosen option: {row['correct_label']}"
        risk_neutral_text = f"{row['prompt_text']}\n\nChosen option: {row['incorrect_label']}"
        
        # Tokenize both inputs
        risk_averse_encoding = self.tokenizer(
            risk_averse_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        risk_neutral_encoding = self.tokenizer(
            risk_neutral_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'risk_averse_input_ids': risk_averse_encoding['input_ids'].flatten(),
            'risk_averse_attention_mask': risk_averse_encoding['attention_mask'].flatten(),
            'risk_neutral_input_ids': risk_neutral_encoding['input_ids'].flatten(),
            'risk_neutral_attention_mask': risk_neutral_encoding['attention_mask'].flatten(),
            'situation_id': row['situation_id']
        }

# Reward model
class RiskAverseRewardModel(nn.Module):
    """Reward model for scoring risk-averse behavior with pairwise ranking loss"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        super().__init__()
        # Disable device_map="auto" on MPS due to compatibility issues
        load_kwargs = {
            "num_labels": 1,
            "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True
        }
        
        # Only use device_map on CUDA, not MPS
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **load_kwargs
        )
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, 
                risk_averse_input_ids=None, risk_averse_attention_mask=None,
                risk_neutral_input_ids=None, risk_neutral_attention_mask=None):
        
        # Handle both single input and pairwise input modes
        if risk_averse_input_ids is not None and risk_neutral_input_ids is not None:
            # Pairwise ranking mode
            return self._forward_pairwise(
                risk_averse_input_ids, risk_averse_attention_mask,
                risk_neutral_input_ids, risk_neutral_attention_mask
            )
        else:
            # Single input mode (for evaluation/inference)
            return self._forward_single(input_ids, attention_mask, labels)
    
    def _forward_single(self, input_ids, attention_mask, labels=None):
        """Standard forward pass for single inputs"""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits.squeeze(-1)
        
        if labels is not None:
            # Use BCE loss for backward compatibility during evaluation
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}
    
    def _forward_pairwise(self, risk_averse_input_ids, risk_averse_attention_mask,
                          risk_neutral_input_ids, risk_neutral_attention_mask):
        """Pairwise ranking forward pass"""
        # Ensure all tensors are on the same device as the model
        device = next(self.backbone.parameters()).device
        
        risk_averse_input_ids = risk_averse_input_ids.to(device)
        risk_averse_attention_mask = risk_averse_attention_mask.to(device)
        risk_neutral_input_ids = risk_neutral_input_ids.to(device)
        risk_neutral_attention_mask = risk_neutral_attention_mask.to(device)
        
        # Get scores for risk-averse choices
        risk_averse_outputs = self.backbone(
            input_ids=risk_averse_input_ids,
            attention_mask=risk_averse_attention_mask
        )
        risk_averse_scores = risk_averse_outputs.logits.squeeze(-1)
        
        # Get scores for risk-neutral choices
        risk_neutral_outputs = self.backbone(
            input_ids=risk_neutral_input_ids,
            attention_mask=risk_neutral_attention_mask
        )
        risk_neutral_scores = risk_neutral_outputs.logits.squeeze(-1)
        
        # Ranking loss: risk-averse should score higher than risk-neutral
        # Using margin ranking loss with margin=1.0
        margin = 1.0
        score_diff = risk_averse_scores - risk_neutral_scores
        
        # Method 1: Standard margin ranking loss
        ranking_loss = torch.relu(margin - score_diff)
        
        # Method 2: Add regularization to prevent scores from becoming too extreme
        score_regularization = 0.01 * (risk_averse_scores.pow(2).mean() + risk_neutral_scores.pow(2).mean())
        
        # Method 3: Ensure there's always some gradient signal
        # Use sigmoid-based loss as fallback when margin loss is zero
        sigmoid_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            score_diff, torch.ones_like(score_diff)
        )
        
        # Combine losses: use margin loss primarily, but add sigmoid loss for gradient flow
        total_loss = ranking_loss.mean() + 0.1 * sigmoid_loss + score_regularization
        
        # Add some debugging info during training
        if self.training:
            avg_risk_averse = risk_averse_scores.mean().item()
            avg_risk_neutral = risk_neutral_scores.mean().item() 
            avg_diff = score_diff.mean().item()
            fraction_wrong = (score_diff < 0).float().mean().item()
            margin_loss_val = ranking_loss.mean().item()
            sigmoid_loss_val = sigmoid_loss.item()
            reg_loss_val = score_regularization.item()
            
            if torch.rand(1).item() < 0.02:  # Print occasionally (2% chance)
                print(f"[DEBUG] RA_avg: {avg_risk_averse:.3f}, RN_avg: {avg_risk_neutral:.3f}, "
                      f"Diff: {avg_diff:.3f}, Wrong: {fraction_wrong:.1%}")
                print(f"[LOSS] Total: {total_loss.item():.3f} = Margin: {margin_loss_val:.3f} + "
                      f"Sigmoid: {sigmoid_loss_val:.3f} + Reg: {reg_loss_val:.3f}")
                print(f"[GRAD] Loss requires_grad: {total_loss.requires_grad}")
                
                # Check if gradients are flowing
                sample_param = next(self.backbone.parameters())
                print(f"[GRAD] Sample param requires_grad: {sample_param.requires_grad}, "
                      f"shape: {sample_param.shape}, mean: {sample_param.mean().item():.6f}")
        
        return {
            "loss": total_loss,
            "risk_averse_scores": risk_averse_scores,
            "risk_neutral_scores": risk_neutral_scores,
            "score_difference": score_diff.mean()
        }

# Training and evaluation functions
# Custom data collator for pairwise training
class PairwiseDataCollator:
    """Data collator for pairwise ranking training"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        # Extract all the different input types
        risk_averse_input_ids = [f['risk_averse_input_ids'] for f in features]
        risk_averse_attention_mask = [f['risk_averse_attention_mask'] for f in features]
        risk_neutral_input_ids = [f['risk_neutral_input_ids'] for f in features]
        risk_neutral_attention_mask = [f['risk_neutral_attention_mask'] for f in features]
        
        # Stack tensors and ensure they're the right type
        batch = {
            'risk_averse_input_ids': torch.stack(risk_averse_input_ids).long(),
            'risk_averse_attention_mask': torch.stack(risk_averse_attention_mask).long(),
            'risk_neutral_input_ids': torch.stack(risk_neutral_input_ids).long(),
            'risk_neutral_attention_mask': torch.stack(risk_neutral_attention_mask).long(),
        }
        
        return batch

def train_reward_model(dataset_df: pd.DataFrame, model_name="Qwen/Qwen2.5-0.5B"):
    """Train the risk-averse reward model with pairwise ranking loss"""
    print(f"Training reward model with {len(dataset_df)} situations using pairwise ranking loss...")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = RiskAverseRewardModel(model_name)
    
    # For MPS compatibility, explicitly move model to CPU if MPS is detected
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("MPS detected but using CPU for better compatibility with pairwise training")
        model = model.cpu()
    
    # Split data
    train_df, val_df = train_test_split(dataset_df, test_size=0.2, random_state=42)
    
    # Create pairwise datasets
    train_dataset = PairwiseRiskAversionDataset(train_df, tokenizer, max_length=128)
    val_dataset = PairwiseRiskAversionDataset(val_df, tokenizer, max_length=128)
    
    print(f"Training on {len(train_dataset)} situation pairs")
    print(f"Validation on {len(val_dataset)} situation pairs")
    
    # Training arguments with memory optimizations
    training_args = TrainingArguments(
        output_dir="./risk_averse_model",
        num_train_epochs=3,
        per_device_train_batch_size=1,        # Reduced from 4 to 1 (now each batch contains 2 inputs per situation)
        per_device_eval_batch_size=1,         # Reduced from 4 to 1
        gradient_accumulation_steps=4,        # Maintain effective batch size of 4
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,                      # Must be multiple of eval_steps for load_best_model_at_end
        load_best_model_at_end=False,        # Disable for pairwise training (no eval_loss available)
        fp16=torch.cuda.is_available(),      # Only enable fp16 on CUDA, not MPS
        dataloader_pin_memory=False,         # Reduce memory transfer overhead
    )
    
    # Custom data collator for pairwise data
    data_collator = PairwiseDataCollator(tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Quick validation check before training
    print("Validating model setup...")
    from torch.utils.data import DataLoader
    sample_batch = next(iter(DataLoader(train_dataset, batch_size=1, collate_fn=data_collator)))
    model.train()
    
    try:
        with torch.no_grad():
            outputs = model(**sample_batch)
            print(f"✓ Model forward pass successful. Loss shape: {outputs['loss'].shape}, Loss value: {outputs['loss'].item():.3f}")
            print(f"✓ Score difference: {outputs['score_difference'].item():.3f}")
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        raise
    
    # Train
    print("Starting pairwise ranking training...")
    trainer.train()
    
    return model, tokenizer, trainer

def evaluate_model(model, tokenizer, test_df: pd.DataFrame, return_detailed=False):
    """Evaluate the trained model"""
    print(f"Evaluating model on {len(test_df)} test situations...")
    
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    risk_averse_wins = 0  # Count how often risk-averse option scores higher
    
    # Store detailed results for plotting
    results = {
        'risk_averse_scores': [],
        'risk_neutral_scores': [],
        'predictions': [],
        'expected': [],
        'situation_ids': []
    }
    
    # Get the device the model is on
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for _, row in test_df.iterrows():
            risk_averse_score = None
            risk_neutral_score = None
            
            # Test both correct and incorrect options using single input mode
            for label, is_correct in [("correct_label", True), ("incorrect_label", False)]:
                chosen_option = row[label]
                input_text = f"{row['prompt_text']}\n\nChosen option: {chosen_option}"
                
                encoding = tokenizer(
                    input_text,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                
                # Move tensors to the same device as model
                encoding = {k: v.to(device) for k, v in encoding.items()}
                
                # Use single input mode for evaluation
                outputs = model(input_ids=encoding['input_ids'], 
                              attention_mask=encoding['attention_mask'])
                
                # Get raw score (no sigmoid needed for ranking comparison)
                raw_score = outputs["logits"].item()
                
                # Store scores for analysis
                if is_correct:  # Risk-averse option
                    risk_averse_score = raw_score
                else:  # Risk-neutral option
                    risk_neutral_score = raw_score
                
                # For accuracy calculation, still use sigmoid and threshold
                prediction = torch.sigmoid(outputs["logits"]).item()
                if (prediction > 0.5) == is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                results['predictions'].append(prediction)
                results['expected'].append(1.0 if is_correct else 0.0)
            
            # Check if risk-averse option scores higher (this is the key metric)
            if risk_averse_score > risk_neutral_score:
                risk_averse_wins += 1
            
            # Store situation-level results
            results['risk_averse_scores'].append(risk_averse_score)
            results['risk_neutral_scores'].append(risk_neutral_score)
            results['situation_ids'].append(row['situation_id'])
    
    accuracy = correct_predictions / total_predictions
    risk_averse_preference_rate = risk_averse_wins / len(test_df)
    
    print(f"Model accuracy: {accuracy:.3f}")
    print(f"Risk-averse preference rate: {risk_averse_preference_rate:.3f}")
    print(f"Average score difference (risk-averse - risk-neutral): {np.mean(np.array(results['risk_averse_scores']) - np.array(results['risk_neutral_scores'])):.3f}")
    
    if return_detailed:
        results['risk_averse_preference_rate'] = risk_averse_preference_rate
        return accuracy, results
    return accuracy

# Plotting functions
def plot_results(trainer, eval_results, accuracy):
    """Create comprehensive plots of training and evaluation results"""
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Risk-Averse Reward Model Training Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    plot_training_loss(trainer, axes[0, 0])
    
    # Plot 2: Score Distribution
    plot_score_distribution(eval_results, axes[0, 1])
    
    # Plot 3: Risk Preference Comparison
    plot_risk_preference_comparison(eval_results, axes[1, 0])
    
    # Plot 4: Model Performance Summary
    plot_performance_summary(eval_results, accuracy, axes[1, 1])
    
    plt.tight_layout()
    
    # Save plots to outputs directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outputs/training_results_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {filename}")
    
    # Show plots if not in headless mode
    try:
        plt.show()
    except:
        print("Display not available - plots saved to file only")

def plot_training_loss(trainer, ax):
    """Plot training and validation loss over time"""
    # Extract loss from trainer logs
    log_history = trainer.state.log_history
    
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []
    
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
    """Plot distribution of model scores for risk-averse vs risk-neutral choices"""
    risk_averse_scores = eval_results['risk_averse_scores']
    risk_neutral_scores = eval_results['risk_neutral_scores']
    
    # Create histogram
    bins = np.linspace(0, 1, 21)
    ax.hist(risk_averse_scores, bins=bins, alpha=0.7, label='Risk-Averse Options', 
            color='green', density=True)
    ax.hist(risk_neutral_scores, bins=bins, alpha=0.7, label='Risk-Neutral Options', 
            color='red', density=True)
    
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, label='Decision Threshold')
    
    ax.set_xlabel('Model Score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution by Option Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
def plot_risk_preference_comparison(eval_results, ax):
    """Plot comparison of scores for risk-averse vs risk-neutral options"""
    risk_averse_scores = np.array(eval_results['risk_averse_scores'])
    risk_neutral_scores = np.array(eval_results['risk_neutral_scores'])
    
    # Scatter plot
    ax.scatter(risk_neutral_scores, risk_averse_scores, alpha=0.6, s=50)
    
    # Add diagonal line (equal preference)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Equal Preference')
    
    # Add preference regions
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.2, color='green', 
                    label='Risk-Averse Preferred')
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.2, color='red', 
                    label='Risk-Neutral Preferred')
    
    ax.set_xlabel('Risk-Neutral Option Score')
    ax.set_ylabel('Risk-Averse Option Score')
    ax.set_title('Risk Preference Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
def plot_performance_summary(eval_results, accuracy, ax):
    """Plot model performance summary statistics"""
    risk_averse_scores = np.array(eval_results['risk_averse_scores'])
    risk_neutral_scores = np.array(eval_results['risk_neutral_scores'])
    
    # Calculate metrics
    correctly_prefers_risk_averse = np.mean(risk_averse_scores > risk_neutral_scores)
    avg_risk_averse_score = np.mean(risk_averse_scores)
    avg_risk_neutral_score = np.mean(risk_neutral_scores)
    score_difference = avg_risk_averse_score - avg_risk_neutral_score
    
    # Create bar plot
    metrics = ['Overall\nAccuracy', 'Risk-Averse\nPreference Rate', 
               'Avg Risk-Averse\nScore', 'Avg Risk-Neutral\nScore']
    values = [accuracy, correctly_prefers_risk_averse, avg_risk_averse_score, avg_risk_neutral_score]
    colors = ['blue', 'green', 'darkgreen', 'darkred']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Summary')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add score difference annotation
    ax.text(0.5, 0.9, f'Score Difference: {score_difference:+.3f}', 
            transform=ax.transAxes, ha='center', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontweight='bold')

# Main experiment function
def run_experiment():
    """Run the complete risk aversion experiment"""
    print("=== Risk-Averse Reward Model Experiment ===")
    
    # Setup environment
    is_colab = setup_environment()
    
    # Load data from CSV
    print("\n1. Loading risk scenario data from CSV...")
    loader = RiskAversionDataLoader()
    full_dataset_df = loader.load_and_process_data()
    
    # Limit to 500 situations for training
    if len(full_dataset_df) > 500:
        dataset_df = full_dataset_df.head(500)
        print(f"Limited dataset to {len(dataset_df)} situations for training")
    else:
        dataset_df = full_dataset_df
        print(f"Using all {len(dataset_df)} available situations")
    
    # Split into train/test
    train_df, test_df = train_test_split(dataset_df, test_size=0.3, random_state=42)
    
    # Train model
    print(f"\n2. Training reward model...")
    model, tokenizer, trainer = train_reward_model(train_df)
    
    # Evaluate
    print(f"\n3. Evaluating model...")
    accuracy, eval_results = evaluate_model(model, tokenizer, test_df, return_detailed=True)
    
    # Plot results
    print(f"\n4. Creating visualizations...")
    plot_results(trainer, eval_results, accuracy)
    
    # Save results to outputs directory
    os.makedirs("outputs", exist_ok=True)
    results = {
        "num_training_situations": len(train_df),
        "num_test_situations": len(test_df),
        "final_accuracy": accuracy,
        "model_name": "Qwen/Qwen2.5-0.5B",
        "timestamp": datetime.now().isoformat()
    }
    
    with open("outputs/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Experiment Complete ===")
    print(f"Results saved to outputs/experiment_results.json")
    print(f"Training plots saved to outputs/ directory with timestamp")
    print(f"Final accuracy: {accuracy:.3f}")
    
    # Print summary statistics
    risk_averse_scores = np.array(eval_results['risk_averse_scores'])
    risk_neutral_scores = np.array(eval_results['risk_neutral_scores'])
    correctly_prefers_risk_averse = np.mean(risk_averse_scores > risk_neutral_scores)
    score_difference = np.mean(risk_averse_scores) - np.mean(risk_neutral_scores)
    
    print(f"Risk-averse preference rate: {correctly_prefers_risk_averse:.3f}")
    print(f"Average score difference (risk-averse - risk-neutral): {score_difference:+.3f}")
    
    return model, tokenizer, results

if __name__ == "__main__":
    # Run the experiment
    try:
        model, tokenizer, results = run_experiment()
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        import traceback
        traceback.print_exc()