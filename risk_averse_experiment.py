"""
Risk-Averse Reward Model Training Experiment

This experiment trains a reward model to be risk-averse using TinyLlama 1.1B model.
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
        # Each row in CSV represents an option, we need to aggregate per situation
        situations = []
        for situation_id, group in df.groupby('situation_id'):
            # Verify all rows in group have same prompt_text, correct_label, incorrect_label
            # (since these should be situation-level properties, not option-level)
            unique_prompts = group['prompt_text'].nunique()
            unique_correct = group['correct_label'].nunique() 
            unique_incorrect = group['incorrect_label'].nunique()
            
            if unique_prompts > 1:
                print(f"Warning: situation_id {situation_id} has {unique_prompts} different prompts, using first")
            if unique_correct > 1:
                print(f"Warning: situation_id {situation_id} has {unique_correct} different correct_labels, using first")
            if unique_incorrect > 1:
                print(f"Warning: situation_id {situation_id} has {unique_incorrect} different incorrect_labels, using first")
            
            # Take the first row for situation-level properties
            first_row = group.iloc[0]
            
            # Process the prompt text - replace thinking instruction with output-only instruction
            prompt_text = first_row['prompt_text']
            if 'You can think about which you\'d choose' in prompt_text:
                prompt_text = prompt_text.replace(
                    'You can think about which you\'d choose and then output your answer.',
                    'Output the label of your chosen option only.'
                )
            
            # Collect all bad answer variations from all rows in the group
            bad_correct_list = []
            bad_incorrect_list = []
            
            for _, row in group.iterrows():
                bad_correct = row.get('bad_correct_answers', '')
                bad_incorrect = row.get('bad_incorrect_answers', '')
                
                if pd.notna(bad_correct) and bad_correct.strip():
                    bad_correct_list.extend([x.strip() for x in str(bad_correct).split(',') if x.strip()])
                if pd.notna(bad_incorrect) and bad_incorrect.strip():
                    bad_incorrect_list.extend([x.strip() for x in str(bad_incorrect).split(',') if x.strip()])
            
            # Remove duplicates and join back
            bad_correct_combined = ','.join(list(set(bad_correct_list))) if bad_correct_list else ''
            bad_incorrect_combined = ','.join(list(set(bad_incorrect_list))) if bad_incorrect_list else ''
            
            situations.append({
                'situation_id': situation_id,
                'prompt_text': prompt_text,
                'correct_label': first_row['correct_label'],
                'incorrect_label': first_row['incorrect_label'],
                'bad_correct_answers': bad_correct_combined,
                'bad_incorrect_answers': bad_incorrect_combined,
                'num_options': len(group)  # Track how many options were in this situation
            })
        
        result_df = pd.DataFrame(situations)
        total_options = df.groupby('situation_id').size().sum()
        print(f"Processed {total_options} option rows into {len(result_df)} unique situations")
        
        # Display sample data
        if len(result_df) > 0:
            sample = result_df.iloc[0]
            print(f"\nSample situation:")
            print(f"Prompt: {sample['prompt_text'][:200]}...")
            print(f"Risk-averse choice: {sample['correct_label']}")
            print(f"Risk-neutral choice: {sample['incorrect_label']}")
            print(f"Options in situation: {sample['num_options']}")
            if sample['bad_correct_answers']:
                print(f"Bad correct variations: {sample['bad_correct_answers']}")
            if sample['bad_incorrect_answers']:
                print(f"Bad incorrect variations: {sample['bad_incorrect_answers']}")
        
        return result_df

# Dataset class for training
class RiskAversionDataset(Dataset):
    """Dataset for risk aversion reward model training"""
    
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length=256):
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
    
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length=256):
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
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        super().__init__()
        # Optimized for CUDA (Colab) with automatic device mapping and fp16
        load_kwargs = {
            "num_labels": 1,
            "dtype": torch.float16,  # Always use fp16 for memory efficiency
            "device_map": "auto",  # Automatic GPU device mapping
            "low_cpu_mem_usage": True,
            "attn_implementation": "flash_attention_2" if torch.cuda.is_available() else None,  # Use Flash Attention if available
        }
        
        # Remove None values
        load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}
            
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
            # Use BCE loss for evaluation mode
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

def train_reward_model(dataset_df: pd.DataFrame, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Train the risk-averse reward model with pairwise ranking loss"""
    print(f"Training reward model with {len(dataset_df)} situations using pairwise ranking loss...")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = RiskAverseRewardModel(model_name)
    
    # Model will be automatically placed on GPU via device_map="auto"
    
    # Split data
    train_df, val_df = train_test_split(dataset_df, test_size=0.2, random_state=42)
    
    # Create pairwise datasets with increased sequence length for larger model
    train_dataset = PairwiseRiskAversionDataset(train_df, tokenizer, max_length=256)
    val_dataset = PairwiseRiskAversionDataset(val_df, tokenizer, max_length=256)
    
    print(f"Training on {len(train_dataset)} situation pairs")
    print(f"Validation on {len(val_dataset)} situation pairs")
    
    # Training arguments with memory optimizations
    training_args = TrainingArguments(
        output_dir="./risk_averse_model",
        num_train_epochs=3,
        per_device_train_batch_size=2,        # Increased batch size for GPU efficiency
        per_device_eval_batch_size=4,         # Larger eval batch for faster evaluation  
        gradient_accumulation_steps=2,        # Maintain effective batch size of 4
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,                      # Must be multiple of eval_steps for load_best_model_at_end
        load_best_model_at_end=False,        # Disable for pairwise training (no eval_loss available)
        fp16=True,                          # Always use fp16 for memory efficiency in Colab
        dataloader_pin_memory=True,          # Enable memory pinning for faster GPU transfer
        dataloader_num_workers=2,           # Use multiple workers for data loading
        remove_unused_columns=False,        # Keep all columns for custom collator
        optim="adamw_torch_fused",          # Use fused AdamW for better GPU utilization
        group_by_length=True,               # Group sequences by length for efficiency
        prediction_loss_only=True,         # Only compute loss (not other metrics)
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

def check_answer_match(model_output: str, correct_answer: str, bad_variations: str) -> bool:
    """Check if model output matches correct answer or any of its bad variations"""
    model_output = model_output.strip()
    
    # Check exact match with correct answer
    if model_output == correct_answer:
        return True
    
    # Check bad variations if they exist
    if bad_variations and pd.notna(bad_variations):
        bad_list = [x.strip() for x in str(bad_variations).split(',') if x.strip()]
        if model_output in bad_list:
            return True
    
    return False

def evaluate_model(model, tokenizer, test_df: pd.DataFrame, return_detailed=False):
    """Evaluate the trained model with support for bad answer variations"""
    print(f"Evaluating model on {len(test_df)} test situations...")
    
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    risk_averse_wins = 0  # Count how often risk-averse option scores higher
    bad_variation_matches = 0  # Count matches via bad variations
    
    # Store detailed results for plotting
    results = {
        'risk_averse_scores': [],
        'risk_neutral_scores': [],
        'predictions': [],
        'expected': [],
        'situation_ids': [],
        'answer_quality_stats': {
            'exact_matches': 0,
            'bad_variation_matches': 0,
            'no_matches': 0
        }
    }
    
    # Get the device the model is on
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for _, row in test_df.iterrows():
            risk_averse_scores = []  # All variations of risk-averse answer
            risk_neutral_scores = []  # All variations of risk-neutral answer
            
            # Test both correct and incorrect options using single input mode
            for label, is_correct, bad_label in [
                ("correct_label", True, "bad_correct_answers"), 
                ("incorrect_label", False, "bad_incorrect_answers")
            ]:
                chosen_option = row[label]
                bad_variations = row.get(bad_label, '')
                
                # Test main answer
                input_text = f"{row['prompt_text']}\n\nChosen option: {chosen_option}"
                encoding = tokenizer(
                    input_text,
                    truncation=True,
                    padding='max_length',
                    max_length=256,
                    return_tensors='pt'
                )
                encoding = {k: v.to(device) for k, v in encoding.items()}
                outputs = model(input_ids=encoding['input_ids'], 
                              attention_mask=encoding['attention_mask'])
                main_score = outputs["logits"].item()
                
                # Test bad variations
                variation_scores = []
                if bad_variations and pd.notna(bad_variations):
                    bad_list = [x.strip() for x in str(bad_variations).split(',') if x.strip()]
                    for bad_answer in bad_list:
                        if bad_answer and bad_answer != chosen_option:  # Avoid duplicates
                            bad_input_text = f"{row['prompt_text']}\n\nChosen option: {bad_answer}"
                            bad_encoding = tokenizer(
                                bad_input_text,
                                truncation=True,
                                padding='max_length',
                                max_length=256,
                                return_tensors='pt'
                            )
                            bad_encoding = {k: v.to(device) for k, v in bad_encoding.items()}
                            bad_outputs = model(input_ids=bad_encoding['input_ids'], 
                                              attention_mask=bad_encoding['attention_mask'])
                            variation_scores.append(bad_outputs["logits"].item())
                
                # Use the highest score among all variations (main + bad)
                all_scores = [main_score] + variation_scores
                best_score = max(all_scores)
                
                # Store scores for analysis
                if is_correct:  # Risk-averse option
                    risk_averse_scores = all_scores
                    risk_averse_score = best_score
                else:  # Risk-neutral option
                    risk_neutral_scores = all_scores
                    risk_neutral_score = best_score
                
                # For accuracy calculation, use best score
                prediction = torch.sigmoid(torch.tensor(best_score)).item()
                if (prediction > 0.5) == is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                results['predictions'].append(prediction)
                results['expected'].append(1.0 if is_correct else 0.0)
                
                # Track if bad variations helped
                if len(variation_scores) > 0 and best_score in variation_scores:
                    bad_variation_matches += 1
            
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
    print(f"Bad variation matches: {bad_variation_matches}/{total_predictions} ({bad_variation_matches/total_predictions:.1%})")
    print(f"Average score difference (risk-averse - risk-neutral): {np.mean(np.array(results['risk_averse_scores']) - np.array(results['risk_neutral_scores'])):.3f}")
    
    # Store bad variation stats
    results['answer_quality_stats']['bad_variation_matches'] = bad_variation_matches
    results['answer_quality_stats']['exact_matches'] = total_predictions - bad_variation_matches
    results['answer_quality_stats']['total_predictions'] = total_predictions
    
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
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
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