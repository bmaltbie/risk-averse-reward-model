# Risk-Averse Reward Model Code Walkthrough

This document provides a detailed, step-by-step walkthrough of the `risk_averse_experiment.py` code, explaining the purpose and reasoning behind each major component.

The current implementation uses pairwise ranking loss to train reward models that prefer risk-averse choices. For historical context and evolution of the codebase, see [CHANGELOG.md](CHANGELOG.md).

## Table of Contents
1. [Imports and Setup](#imports-and-setup)
2. [Environment Detection](#environment-detection)
3. [Data Loading](#data-loading)
4. [Dataset Classes](#dataset-classes)
5. [Model Architecture](#model-architecture)
6. [Training Pipeline](#training-pipeline)
7. [Evaluation System](#evaluation-system)
8. [Visualization Components](#visualization-components)
9. [Main Experiment Flow](#main-experiment-flow)

---

## Imports and Setup

```python
import os, sys, pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                         TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.model_selection import train_test_split
import json, warnings, matplotlib.pyplot as plt, seaborn as sns
from datetime import datetime
```

**Purpose**: Import all necessary libraries for:
- **PyTorch**: Deep learning framework for model training
- **Transformers**: Hugging Face library for pre-trained language models
- **Pandas/NumPy**: Data manipulation and numerical operations
- **Matplotlib/Seaborn**: Visualization and plotting
- **Scikit-learn**: Data splitting utilities

**Styling Setup**:
```python
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')
```

**Reasoning**: Sets consistent visual styling for plots and suppresses non-critical warnings that might clutter output during training.

---

## Environment Detection

```python
def setup_environment():
    try:
        import google.colab
        IN_COLAB = True
        print("Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("Running locally")
    
    if IN_COLAB:
        os.system("pip install transformers datasets accelerate")
    
    return IN_COLAB
```

**Purpose**: Automatically detect whether the code is running in Google Colab or locally.

**Reasoning**: 
- **Colab Detection**: Tries to import `google.colab` module - only available in Colab environments
- **Automatic Setup**: Installs required packages in Colab since they're not pre-installed
- **Environment Awareness**: Allows code to adapt behavior based on running environment

---

## Data Loading

### RiskAversionDataLoader Class

```python
class RiskAversionDataLoader:
    def __init__(self, csv_file_path="data/strict_disagreements_10k_with_prompts_and_bad_formats.csv"):
        self.csv_file_path = csv_file_path
```

**Purpose**: Handles loading and preprocessing of risk scenario data from CSV files.

### Core Loading Logic

```python
def load_and_process_data(self) -> pd.DataFrame:
    # Check if CSV file exists
    if not os.path.exists(self.csv_file_path):
        raise FileNotFoundError(f"Required data file '{self.csv_file_path}' not found.")
    
    # Load the CSV file
    df = pd.read_csv(self.csv_file_path)
    print(f"Loaded {len(df)} rows from {self.csv_file_path}")
```

**Reasoning**: 
- **Fail Fast**: Immediately fails if required data file is missing rather than continuing with fallback data
- **Transparency**: Reports exactly how many rows were loaded for debugging

### Data Validation

```python
# Check required columns exist
required_columns = ['situation_id', 'prompt_text', 'correct_label', 'incorrect_label']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in CSV: {missing_columns}")
```

**Purpose**: Validates that the CSV contains all necessary columns for training.

**Reasoning**: Data validation prevents cryptic errors later in the pipeline by catching format issues early.

### Data Grouping and Processing

```python
# Group by situation_id to get unique situations
situations = []
for situation_id, group in df.groupby('situation_id'):
    first_row = group.iloc[0]  # Take first row for each situation
    
    # Process the prompt text - replace thinking instruction with output-only instruction
    prompt_text = first_row['prompt_text']
    if 'You can think about which you\'d choose' in prompt_text:
        prompt_text = prompt_text.replace(
            'You can think about which you\'d choose and then output your answer.',
            'Output the label of your chosen option only.'
        )
```

**Purpose**: 
- **Deduplication**: Groups multiple rows with same `situation_id` into single scenarios
- **Prompt Modification**: Changes prompts from thinking-based to output-only format

**Reasoning**: 
- **Data Structure**: Original CSV may have multiple rows per situation (one per option), but we need one row per situation for training
- **Prompt Engineering**: "Output only" format is better for reward model training than allowing reasoning, as we want to score the final choice

---

## Dataset Classes

### PairwiseRiskAversionDataset

```python
class PairwiseRiskAversionDataset(Dataset):
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
        risk_averse_encoding = self.tokenizer(risk_averse_text, ...)
        risk_neutral_encoding = self.tokenizer(risk_neutral_text, ...)
        
        return {
            'risk_averse_input_ids': risk_averse_encoding['input_ids'].flatten(),
            'risk_averse_attention_mask': risk_averse_encoding['attention_mask'].flatten(),
            'risk_neutral_input_ids': risk_neutral_encoding['input_ids'].flatten(),
            'risk_neutral_attention_mask': risk_neutral_encoding['attention_mask'].flatten(),
            'situation_id': row['situation_id']
        }
```

**Purpose**: Pairwise dataset that provides both risk-averse and risk-neutral choices for direct comparison.

**Key Design Advantages**:
- **Direct Comparison**: Each training example contains both options from the same scenario
- **Efficient Training**: One example per situation
- **Ranking Optimization**: Designed for pairwise ranking loss training
- **Clear Structure**: Explicit separation of risk-averse vs risk-neutral inputs

### Custom Data Collator

```python
class PairwiseDataCollator:
    def __call__(self, features):
        # Extract all the different input types
        risk_averse_input_ids = [f['risk_averse_input_ids'] for f in features]
        risk_neutral_input_ids = [f['risk_neutral_input_ids'] for f in features]
        # ... similar for attention masks
        
        # Stack tensors and ensure they're the right type
        batch = {
            'risk_averse_input_ids': torch.stack(risk_averse_input_ids).long(),
            'risk_neutral_input_ids': torch.stack(risk_neutral_input_ids).long(),
            # ... similar for attention masks
        }
        return batch
```

**Purpose**: Custom collator that properly batches pairwise inputs for ranking training.

**Reasoning**:
- **Tensor Management**: Handles 4 separate tensor types (2 inputs × 2 tensor types each)
- **Type Safety**: Explicit `.long()` conversion for token IDs (required for MPS compatibility)
- **Batch Structure**: Creates properly structured batches for pairwise forward pass

### Tokenization

```python
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
```

**Purpose**: Converts text to token IDs that the model can process.

**Reasoning**:
- **Fixed Length**: `max_length=128` provides consistent tensor sizes for efficient batching
- **Truncation**: Handles edge cases where prompts exceed max length
- **Padding**: Ensures all sequences are same length
- **Float Labels**: Uses 1.0/0.0 for risk-averse/risk-neutral to match sigmoid output

---

## Model Architecture

### RiskAverseRewardModel

```python
class RiskAverseRewardModel(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        super().__init__()
        
        # Conditional loading based on device capabilities
        load_kwargs = {
            "num_labels": 1,
            "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True
        }
        
        # Only use device_map on CUDA, not MPS
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name, **load_kwargs
        )
```

**Purpose**: Wrapper around pre-trained language model for reward scoring.

**Memory Optimization Reasoning**:
- **Conditional Precision**: Uses fp16 on CUDA for memory savings, fp32 on CPU/MPS for compatibility
- **Device Mapping**: Only uses automatic device mapping on CUDA where it's fully supported
- **Low CPU Usage**: Reduces CPU memory during model loading

### Forward Pass

```python
def forward(self, input_ids, attention_mask, labels=None):
    outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits.squeeze(-1)
    
    if labels is not None:
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}
    
    return {"logits": logits}
```

**Purpose**: Defines how the model processes inputs and computes loss.

**Reasoning**:
- **Single Output**: `num_labels=1` and `squeeze(-1)` produces scalar reward scores
- **BCE Loss**: Binary Cross-Entropy loss used for evaluation mode
- **Conditional Loss**: Only computes loss during training, not inference

---

## Training Pipeline

### Training Function Setup

```python
def train_reward_model(dataset_df: pd.DataFrame, model_name="Qwen/Qwen2.5-0.5B"):
    print(f"Training reward model with {len(dataset_df)} situations...")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = RiskAverseRewardModel(model_name)
```

**Pad Token Fix**: Many models don't have a dedicated padding token, so we use the end-of-sequence token as a fallback.

### Data Splitting

```python
# Split data
train_df, val_df = train_test_split(dataset_df, test_size=0.2, random_state=42)

# Create datasets with reduced max_length
train_dataset = RiskAversionDataset(train_df, tokenizer, max_length=128)
val_dataset = RiskAversionDataset(val_df, tokenizer, max_length=128)
```

**Purpose**: Creates 80/20 train/validation split for monitoring overfitting.

### Training Arguments

```python
training_args = TrainingArguments(
    output_dir="./risk_averse_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,        # Memory optimization
    per_device_eval_batch_size=1,         
    gradient_accumulation_steps=4,        # Maintain effective batch size
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,                      # Must match eval_steps for best model loading
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=torch.cuda.is_available(),      # Platform-specific optimization
    dataloader_pin_memory=False,         # Reduce memory overhead
)
```

**Memory Optimization Strategy**:
- **Small Batches**: `batch_size=1` minimizes memory per step
- **Gradient Accumulation**: `steps=4` maintains effective batch size of 4 for stable training
- **Synchronized Steps**: `save_steps=eval_steps` required for best model loading

**Reasoning**:
- **Warmup**: Gradually increases learning rate for stable training start
- **Weight Decay**: L2 regularization prevents overfitting
- **Best Model**: Automatically loads the checkpoint with lowest validation loss

---

## Evaluation System

### Enhanced Evaluation Function

```python
def evaluate_model(model, tokenizer, test_df: pd.DataFrame, return_detailed=False):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    # Store detailed results for plotting
    results = {
        'risk_averse_scores': [],
        'risk_neutral_scores': [],
        'predictions': [],
        'expected': [],
        'situation_ids': []
    }
```

**Purpose**: Evaluates model performance and optionally returns detailed results for visualization.

### Evaluation Logic

```python
device = next(model.parameters()).device

with torch.no_grad():
    for _, row in test_df.iterrows():
        risk_averse_score = None
        risk_neutral_score = None
        
        # Test both correct and incorrect options
        for label, is_correct in [("correct_label", True), ("incorrect_label", False)]:
            chosen_option = row[label]
            input_text = f"{row['prompt_text']}\n\nChosen option: {chosen_option}"
            
            encoding = tokenizer(input_text, truncation=True, padding='max_length', 
                               max_length=128, return_tensors='pt')
            
            # Move tensors to same device as model
            encoding = {k: v.to(device) for k, v in encoding.items()}
            
            outputs = model(**encoding)
            prediction = torch.sigmoid(outputs["logits"]).item()
```

**Device Management**: Ensures input tensors are on same device as model (crucial for MPS/CUDA compatibility).

**Sigmoid Conversion**: Converts raw logits to probabilities between 0 and 1 for interpretable scores.

### Accuracy Calculation

```python
# Check if prediction aligns with expected reward
if (prediction > 0.5) == is_correct:
    correct_predictions += 1
total_predictions += 1

# Store scores for analysis
if is_correct:  # Risk-averse option
    risk_averse_score = prediction
else:  # Risk-neutral option
    risk_neutral_score = prediction
```

**Binary Classification**: Uses 0.5 threshold to convert probability scores to binary decisions.

**Detailed Tracking**: Stores individual scores for each option type to enable rich analysis and visualization.

---

## Visualization Components

### Main Plotting Function

```python
def plot_results(trainer, eval_results, accuracy):
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Risk-Averse Reward Model Training Results', fontsize=16, fontweight='bold')
```

**Purpose**: Creates comprehensive 4-panel visualization of training results.

### Training Loss Plot

```python
def plot_training_loss(trainer, ax):
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
```

**Purpose**: Extracts training metrics from Hugging Face trainer logs.

**Reasoning**: Training and validation loss curves help identify overfitting and training convergence.

### Score Distribution Analysis

```python
def plot_score_distribution(eval_results, ax):
    risk_averse_scores = eval_results['risk_averse_scores']
    risk_neutral_scores = eval_results['risk_neutral_scores']
    
    bins = np.linspace(0, 1, 21)
    ax.hist(risk_averse_scores, bins=bins, alpha=0.7, label='Risk-Averse Options', 
            color='green', density=True)
    ax.hist(risk_neutral_scores, bins=bins, alpha=0.7, label='Risk-Neutral Options', 
            color='red', density=True)
    
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, label='Decision Threshold')
```

**Purpose**: Visualizes how the model scores different option types.

**Reasoning**: 
- **Distribution Overlap**: Shows how well the model separates risk-averse from risk-neutral choices
- **Decision Threshold**: 0.5 line shows the classification boundary
- **Density Normalization**: Enables comparison even with different sample sizes

### Risk Preference Scatter Plot

```python
def plot_risk_preference_comparison(eval_results, ax):
    risk_averse_scores = np.array(eval_results['risk_averse_scores'])
    risk_neutral_scores = np.array(eval_results['risk_neutral_scores'])
    
    ax.scatter(risk_neutral_scores, risk_averse_scores, alpha=0.6, s=50)
    
    # Add diagonal line (equal preference)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Equal Preference')
    
    # Add preference regions
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.2, color='green', 
                    label='Risk-Averse Preferred')
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.2, color='red', 
                    label='Risk-Neutral Preferred')
```

**Purpose**: Shows relationship between scores for both option types in each scenario.

**Reasoning**:
- **Diagonal Analysis**: Points above diagonal indicate risk-averse preference
- **Preference Regions**: Colored areas show which type of choice the model prefers
- **Scatter Pattern**: Reveals consistency of model preferences across scenarios

### Performance Summary

```python
def plot_performance_summary(eval_results, accuracy, ax):
    risk_averse_scores = np.array(eval_results['risk_averse_scores'])
    risk_neutral_scores = np.array(eval_results['risk_neutral_scores'])
    
    # Calculate metrics
    correctly_prefers_risk_averse = np.mean(risk_averse_scores > risk_neutral_scores)
    avg_risk_averse_score = np.mean(risk_averse_scores)
    avg_risk_neutral_score = np.mean(risk_neutral_scores)
    score_difference = avg_risk_averse_score - avg_risk_neutral_score
```

**Purpose**: Computes and displays key performance metrics.

**Key Metrics**:
- **Overall Accuracy**: Binary classification performance
- **Risk-Averse Preference Rate**: How often model prefers risk-averse options
- **Average Scores**: Mean scores for each option type
- **Score Difference**: Quantitative measure of bias toward risk aversion

---

## Main Experiment Flow

### Experiment Orchestration

```python
def run_experiment():
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
```

**Dataset Limiting**: Restricts training to 500 situations for faster experimentation while maintaining meaningful sample size.

### Training and Evaluation

```python
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
```

**Flow Logic**:
1. **70/30 Split**: Reserves larger portion for evaluation than validation (30% vs 20%)
2. **Sequential Execution**: Each step builds on results from the previous step
3. **Comprehensive Output**: Combines training logs, evaluation metrics, and visualizations

### Results Saving

```python
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
```

**Purpose**: Saves experiment metadata for reproducibility and comparison across runs.

### Summary Statistics

```python
# Print summary statistics
risk_averse_scores = np.array(eval_results['risk_averse_scores'])
risk_neutral_scores = np.array(eval_results['risk_neutral_scores'])
correctly_prefers_risk_averse = np.mean(risk_averse_scores > risk_neutral_scores)
score_difference = np.mean(risk_averse_scores) - np.mean(risk_neutral_scores)

print(f"Risk-averse preference rate: {correctly_prefers_risk_averse:.3f}")
print(f"Average score difference (risk-averse - risk-neutral): {score_difference:+.3f}")
```

**Purpose**: Provides immediate feedback on model performance with key interpretable metrics.

---

## Error Handling and Robustness

### Exception Management

```python
if __name__ == "__main__":
    try:
        model, tokenizer, results = run_experiment()
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        import traceback
        traceback.print_exc()
```

**Purpose**: Graceful handling of interruptions and errors with helpful debugging information.

### Device Compatibility

The code includes multiple layers of device compatibility:
- **Conditional fp16**: Only on CUDA where fully supported
- **Device mapping**: Only on CUDA, not MPS
- **Explicit tensor movement**: Ensures inputs match model device
- **Platform detection**: Adapts behavior for Colab vs local execution

**Reasoning**: Ensures code runs across different hardware configurations (CPU, CUDA GPU, Apple Silicon MPS) without modification.

---

## Summary

This code implements a complete pipeline for training a reward model to prefer risk-averse choices:

1. **Data Loading**: Robust CSV processing with validation and prompt modification
2. **Model Architecture**: Memory-optimized transformer with reward scoring head
3. **Training**: Efficient training with gradient accumulation and best model selection
4. **Evaluation**: Comprehensive scoring of both option types per scenario
5. **Visualization**: Rich plots showing training progress and model behavior
6. **Organization**: Clean outputs directory structure with timestamped results

The design prioritizes memory efficiency, cross-platform compatibility, and thorough analysis while maintaining code clarity and robustness.

---

## Technical Architecture and Implementation Narrative

The risk-averse reward model represents a carefully architected solution using pairwise ranking optimization. At its core, the system addresses a fundamental challenge in AI alignment: training models to exhibit human-like risk preferences rather than maximizing expected value alone.

**Architectural Philosophy**

The implementation uses pairwise ranking loss to directly optimize the core objective: ensuring risk-averse choices consistently score higher than risk-neutral alternatives. This approach avoids spurious correlation learning by comparing options within the same scenario context.

The RiskAverseRewardModel embodies this dual-mode philosophy through its flexible forward pass architecture. In pairwise training mode, it simultaneously processes both risk-averse and risk-neutral inputs from the same scenario, computing their respective scores and applying a hybrid loss function that combines margin ranking loss, sigmoid-based gradient flow, and L2 regularization. This multi-component loss design ensures robust optimization even when margin violations are sparse, while preventing score explosion through regularization. In single evaluation mode, the model supports individual option scoring for evaluation workflows, enabling standard metric computation.

**Data Pipeline and Processing Strategy**

The data architecture reflects careful consideration of both training efficiency and semantic correctness. The PairwiseRiskAversionDataset presents complete scenario contexts with both choices explicitly tokenized, enabling the model to learn genuine preference patterns within scenario contexts.

The custom PairwiseDataCollator handles the complex tensor management required for dual-input training, ensuring proper batching of four distinct tensor streams (risk-averse and risk-neutral inputs, each with their respective attention masks) while maintaining type safety across different hardware platforms. The prompt engineering strategy—replacing thinking-based instructions with output-only commands—optimizes for reward model training by focusing on final choice evaluation rather than reasoning processes.

**Memory Optimization and Hardware Compatibility**

Memory efficiency considerations permeate every architectural decision. The choice of Qwen2.5-0.5B over larger models, sequence length reduction to 128 tokens, and gradient accumulation with minimal batch sizes collectively enable training on resource-constrained environments while maintaining training stability. The conditional precision strategy—fp16 on CUDA, fp32 elsewhere—maximizes hardware utilization while avoiding compatibility issues.

The multi-platform compatibility layer represents sophisticated device management, with automatic fallback strategies for Apple Silicon's MPS implementation. The explicit CPU fallback for pairwise training on MPS devices addresses specific PyTorch limitations with placeholder storage during complex tensor operations, ensuring consistent behavior across development environments.

**Loss Function Design and Training Dynamics**

The hybrid loss function architecture represents the system's most sophisticated component. The margin ranking loss provides the primary optimization signal, encouraging score separation between risk-averse and risk-neutral choices. The sigmoid component ensures continuous gradient flow even when margin constraints are satisfied, preventing training stagnation. The regularization term prevents score explosion while maintaining reasonable value ranges.

The real-time debugging integration provides visibility into training dynamics, displaying score distributions, preference rates, and loss component breakdowns during training. This capability enables monitoring convergence and validating that the model learns genuine risk preferences.

**Evaluation Philosophy and Metrics**

The evaluation system shifts focus from traditional classification accuracy to ranking-based metrics that directly measure the training objective. The risk-averse preference rate—the fraction of scenarios where risk-averse options score higher—serves as the primary success metric. Score difference analysis provides quantitative measures of preference strength, while detailed per-situation tracking enables fine-grained analysis of model behavior patterns.

**Visualization and Analysis Framework**

The comprehensive plotting system transforms raw training data into interpretable insights through four complementary perspectives: training progress monitoring, score distribution analysis, preference comparison visualization, and performance summary statistics. The scatter plot analysis—plotting risk-neutral versus risk-averse scores with preference region highlighting—provides immediate visual feedback on model learning progress.

**Production Readiness and Extensibility**

The modular architecture enables straightforward extension to additional risk metrics or preference types. The clean separation between data loading, model architecture, training pipeline, and evaluation components allows independent modification of each subsystem. The comprehensive error handling and device compatibility layers ensure robust deployment across diverse hardware configurations.

The system's design reflects deep understanding of both the theoretical foundations of preference learning and the practical constraints of model training infrastructure. By combining sophisticated loss function design with careful attention to memory efficiency and cross-platform compatibility, it provides a robust foundation for risk-averse AI research while maintaining the flexibility needed for future enhancements and extensions.