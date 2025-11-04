# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project focused on training risk-averse reward models for LLMs. The goal is to train language models (TinyLlama 1.1B) to prefer risk-averse choices over risk-neutral ones using pairwise ranking loss and data from `strict_disagreements_10k_with_prompts_and_bad_formats.csv`.

**Optimized for Google Colab with CUDA GPU support.**

For project evolution and historical context, see [CHANGELOG.md](CHANGELOG.md).

## Key Files

- `risk_averse_experiment.py` - Main experiment implementation with CSV data loading, model training, and evaluation
- `test_experiment.py` - Quick end-to-end test with minimal data and training parameters
- `data/test_data.csv` - Minimal test data (5 scenarios) for quick testing
- `colab_notebook.ipynb` - Jupyter notebook version for Google Colab execution
- `data/strict_disagreements_10k_with_prompts_and_bad_formats.csv` - Training data with risk scenarios (required for full experiment)
- `outputs/` - Directory containing all experiment outputs (plots, results JSON)
- `requirements.txt` - Python dependencies (includes matplotlib and seaborn for plotting)
- `README.md` - Project background and goals

## Development Commands

### Google Colab (Recommended)
- Upload `colab_notebook.ipynb` to Colab
- **Enable GPU**: Runtime → Change runtime type → Hardware accelerator → GPU
- For quick test: Upload `test_data.csv` and run test cells
- For full experiment: Upload `strict_disagreements_10k_with_prompts_and_bad_formats.csv` to Colab
- Run cells sequentially
- Dependencies are installed automatically via `!pip install`
- **Optimized for T4/V100 GPUs** with 16GB+ VRAM

### Local Testing (Limited Support)

For development only - main execution should be on Colab:

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test with minimal data (5 scenarios, 1 epoch)
python test_experiment.py
```

**Note**: Local execution is not optimized and may have memory limitations.

## Visualization Features

The experiment automatically generates comprehensive plots:

### 1. Training Progress
- **Training Loss**: Shows loss reduction over training steps
- **Validation Loss**: Tracks overfitting and model convergence

### 2. Score Analysis
- **Score Distribution**: Histograms comparing risk-averse vs risk-neutral option scores
- **Decision Threshold**: Shows 0.5 threshold line for classification

### 3. Risk Preference Analysis
- **Scatter Plot**: Risk-averse scores vs risk-neutral scores for each scenario
- **Preference Regions**: Visual areas showing which risk preference the model exhibits
- **Diagonal Line**: Equal preference baseline

### 4. Performance Summary
- **Overall Accuracy**: Binary classification accuracy
- **Risk-Averse Preference Rate**: How often model prefers risk-averse options
- **Average Scores**: Mean scores for both option types
- **Score Difference**: Quantitative measure of risk aversion bias

### Plot Output
- All plots saved to `outputs/` directory for organization
- High-resolution PNG files with timestamps
- Filename format: `outputs/training_results_YYYYMMDD_HHMMSS.png`
- Experiment results also saved to `outputs/experiment_results.json`
- Automatic display if GUI available, file-only otherwise

## Architecture

### Core Components

1. **RiskAversionDataLoader**: Loads and processes data from CSV file
   - Reads `strict_disagreements_10k_with_prompts_and_bad_formats.csv` containing pre-generated scenarios
   - Groups data by situation_id to get unique scenarios
   - Modifies prompts to replace thinking instructions with output-only instructions
   - Fails with clear error message if CSV file is missing or has wrong format

2. **PairwiseRiskAversionDataset**: PyTorch dataset for ranking loss training
   - Each scenario generates 1 training pair: risk-averse vs risk-neutral choice
   - Tokenizes both options simultaneously for direct comparison
   - Uses **PairwiseDataCollator** for proper batching of paired inputs

3. **RiskAverseRewardModel**: Reward model with dual forward modes
   - **Pairwise mode**: Takes both risk-averse and risk-neutral inputs, optimizes ranking loss
   - **Single mode**: Standard evaluation mode for individual option scoring
   - **Hybrid Loss**: Combines margin ranking, sigmoid, and L2 regularization components
   - Optimizes for risk-averse choices scoring higher than risk-neutral ones

### Training Pipeline
- CSV data loading → Limit to 500 situations → Train/validation split → Tokenization → Transformer fine-tuning → Evaluation → Visualization
- Uses Hugging Face Transformers with custom dataset and model classes
- **Colab GPU-optimized configuration**:
  - 1.1B TinyLlama model for efficient performance with GPU memory
  - 256 token sequences for improved context understanding
  - Limited to 500 training situations for faster experimentation
  - Gradient accumulation with larger batches for GPU efficiency
  - Flash Attention and fused optimizers when available
- Automatic plotting and visualization of training progress and results
- Requires CSV file to be present - experiment will fail if data file is missing

## Model Configuration

- Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (optimized for Colab GPU performance)
- Risk-averse utility function: `u(w) = 1 - e^(-0.01w)`
- **Training uses Pairwise Ranking Loss** to directly optimize risk-averse preference
- Hybrid loss combining margin ranking, sigmoid, and L2 regularization
- **GPU Optimizations for Colab**:
  - Always uses fp16 mixed precision for memory efficiency
  - Automatic device mapping with `device_map="auto"`
  - Flash Attention 2 when available
  - Fused AdamW optimizer (`adamw_torch_fused`)
  - Sequence length: 256 tokens for better context
  - Batch size: 2 with gradient accumulation (2 steps)
  - Memory pinning enabled for faster GPU transfers
- Comprehensive visualization:
  - Training/validation loss curves
  - Score distribution histograms
  - Risk preference scatter plots
  - Performance summary metrics

## Data Format

Each scenario from the CSV includes:
- `situation_id`: Unique identifier for grouping related options
- `prompt_text`: Text describing the decision scenario (modified to use output-only instructions)
- `correct_label`: Option preferred by risk-averse agent (higher utility)
- `incorrect_label`: Option preferred by risk-neutral agent (higher expected value)
- `bad_correct_answers`: Variations of correct label (e.g., 'a' vs 'A')
- `bad_incorrect_answers`: Variations of incorrect label

## Data Files

### Test Data
- `data/test_data.csv` - Minimal test data with 5 scenarios for quick testing
- Always available, used by `test_experiment.py`

### Full Experiment Data
The full experiment **requires** `data/strict_disagreements_10k_with_prompts_and_bad_formats.csv` to be present. The experiment will fail with a clear error message if this file is missing or doesn't contain the required columns:

**Required columns:**
- `situation_id`: Unique identifier for grouping
- `prompt_text`: Decision scenario text
- `correct_label`: Risk-averse preferred option
- `incorrect_label`: Risk-neutral preferred option

**Optional columns:**
- `bad_correct_answers`: Alternative formats for correct label
- `bad_incorrect_answers`: Alternative formats for incorrect label