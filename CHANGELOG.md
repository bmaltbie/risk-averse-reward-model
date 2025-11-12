# Changelog

All notable changes to the risk-averse reward model project will be documented in this file.

## [2.7.0] - Switch to Low-Stakes Training Dataset - 2025-11-11

### Major Change: New Dataset with Explicit CARA vs LINEAR Labels

**Switching from 10K mixed-stakes situations to 1,000 low-stakes situations with explicit utility markers.**

### The New Dataset

**File**: `data/11_7_low_stakes_training_set.csv`

Key characteristics:
- **1,000 unique situations** (vs 10,000 in previous dataset)
- **Low-stakes gambles**: $0-$100 range monetary outcomes
- **Explicit utility labeling**:
  - `is_best_cara = True`: Marks risk-averse option (CARA utility maximizer)
  - `is_best_linear = True`: Marks risk-neutral option (LINEAR utility maximizer)
- **Multi-row format**: Each situation has 2-5 option rows
- **0-indexed options**: `option_index` in data, but prompts use 1-indexed numbers

### Format Changes

**Old format** (`strict_disagreements_10k_with_prompts_and_bad_formats.csv`):
```
situation_id | prompt_text | correct_label | incorrect_label | bad_correct_answers | bad_incorrect_answers
```
- One row per situation
- Labels were option numbers/letters
- Included bad answer variations

**New format** (`11_7_low_stakes_training_set.csv`):
```
situation_id | option_index | prompt_text | is_best_cara | is_best_linear | num_options | outcomes | probs_percent | EU_cara | EU_linear
```
- Multiple rows per situation (one per option)
- Explicit utility markers instead of pre-labeled correct/incorrect
- No bad answer variations
- Includes utility calculations for verification

### Changes

**Cell 0 (Header Documentation):**
- Updated CSV filename: `strict_disagreements_10k_with_prompts_and_bad_formats.csv` → `11_7_low_stakes_training_set.csv`
- Training scale: 10,000 → 1,000 situations
- Training examples: 20,000 → 2,000
- Training steps: ~25,000 → ~2,500
- Training time: 4-6 hours → 30-60 minutes on A100
- Added "CARA vs LINEAR Utility" feature description

**Cell 6 (RiskAversionDataLoader):**
- Complete rewrite to handle multi-row-per-situation format
- Default CSV path: `11_7_low_stakes_training_set.csv`
- New logic:
  - Groups by `situation_id`
  - Finds row where `is_best_cara == True` (risk-averse option)
  - Finds row where `is_best_linear == True` (risk-neutral option)
  - Converts 0-indexed `option_index` to 1-indexed option numbers
- Removed: `bad_correct_answers` / `bad_incorrect_answers` handling
- Added: Skips situations missing CARA or LINEAR best options (though all 1,000 have both)

**Cell 9 (Model Documentation):**
- Updated training scale table (1,000 situations, 2,000 examples, ~2,500 steps)
- Added "Dataset: Low-Stakes Gambles" section explaining new data format
- Added "Why Low Stakes?" section explaining benefits
- Updated training distribution numbers

**Cell 12 (Evaluation Function):**
- Removed all `bad_correct_answers` / `bad_incorrect_answers` handling
- Simplified to test only main options (no variations)
- Removed `bad_variation_matches` counter
- Progress reporting: every 25 → every 50 situations (smaller dataset)
- Cleaner, more readable code (~50% fewer lines)

**Cell 16 (Training Function):**
- Adjusted hyperparameters for smaller dataset:
  - Warmup steps: 500 → 100
  - Logging steps: 100 → 50 (more frequent)
  - Eval steps: 1000 → 250 (more frequent)
  - Save steps: 1000 → 250
- Training time estimate: 4-6 hours → 30-60 minutes

**Cell 18 (Experiment Function):**
- Updated to reference new CSV filename in printout
- Added `"dataset": "11_7_low_stakes_training_set.csv"` to results JSON
- Updated comments and documentation strings

**Cell 19-20 (Execution Cells):**
- Updated file references in markdown and error messages
- Updated to mention new CSV filename

### Rationale

**Why switch to low-stakes data?**

1. **Simpler numerical patterns**: $0-$100 range is easier for the model to process than mixed stakes ranging from cents to millions
2. **More consistent risk behavior**: Low-stakes gambles show clearer CARA vs LINEAR utility differences
3. **Explicit labeling**: `is_best_cara` and `is_best_linear` flags make data provenance transparent
4. **Faster iteration**: 1,000 situations = 30-60 min training instead of 4-6 hours
5. **Verification data**: Includes `EU_cara` and `EU_linear` columns for validation
6. **No format ambiguity**: No `bad_correct_answers` variations to handle

### Data Validation

**Verified**:
- ✓ All 1,000 situations have both CARA and LINEAR best options
- ✓ Options correctly converted from 0-indexed to 1-indexed
- ✓ No situations skipped (perfect data quality)
- ✓ Prompts use 1-indexed numbering matching our labels

**Example transformation**:
```python
# Raw data (0-indexed)
option_index = 0, is_best_cara = True   # First option (index 0)
option_index = 1, is_best_linear = True  # Second option (index 1)

# Processed (1-indexed for prompts)
correct_label = "1"    # Risk-averse option (shown as "1" in prompt)
incorrect_label = "2"  # Risk-neutral option (shown as "2" in prompt)
```

### Expected Results

With cleaner, lower-stakes data:
- Should see same or better learning as previous attempts
- Faster experimentation cycle (30-60 min vs 4-6 hours)
- More transparent data provenance
- Easier debugging with explicit utility columns

### Testing

Tested `RiskAversionDataLoader` with new CSV:
- ✓ Successfully loaded 3,000 rows
- ✓ Processed to 1,000 unique situations
- ✓ All required columns present
- ✓ Option labels correctly formatted as strings ("1", "2", etc.)
- ✓ Sample outputs show correct CARA/LINEAR mapping

## [2.6.0] - Scale Up: Llama-3-8B with 10K Situations and 10 Epochs - 2025-11-04

### Major Change: Massive Scale-Up to Test Task Learnability

**After discovering the task requires learning CARA utility function, scaling up to test if it's learnable with more capacity and data.**

### The Discovery

Analysis of 100 situations revealed:
- **98% EV consistency**: Risk-neutral options have higher expected value ✓
- **78% variance consistency**: Risk-averse options have lower variance ✗

This inconsistency revealed the labels are based on **CARA (Constant Absolute Risk Aversion) utility**:
```
u(w) = 1 - e^(-0.01w)
```

Not simple variance! The model must implicitly learn this exponential utility function from examples alone.

### Why Previous Experiments Failed

**TinyLlama 1.1B + 350 situations + 3 epochs was insufficient** because:
1. **Task complexity**: Must learn graduate-level economics (CARA utility)
2. **Insufficient capacity**: 1.1B parameters can't learn complex mathematical patterns
3. **Too little data**: 350 situations not enough to discover the utility function
4. **Too few epochs**: 3 epochs provides limited exposure to each example

### The Scale-Up Solution

Scaling up **71× in total training** to test if task is learnable:

| Parameter | Previous | **New** | Multiplier |
|-----------|----------|---------|------------|
| Model | TinyLlama 1.1B | **Llama-3-8B** | 7× parameters |
| Situations | 350 | **10,000** | 28× more data |
| Examples | 700 | **20,000** | 28× |
| Epochs | 3 | **10** | 3× |
| Total steps | ~350 | **~25,000** | **71×** |
| Training time | 5 min | **4-6 hours** | 48-72× |

### Changes

**Cell 0 (Header):**
- Updated to reflect Llama-3-8B and scaled training
- Added A100 GPU requirement
- Documented 4-6 hour training time

**Cell 9 (Model Documentation):**
- Complete comparison table showing scale-up
- Explanation of why CARA utility requires this scale
- Memory optimization details for 8B model
- Expected behavior with larger model

**Cell 10 (RiskAverseRewardModel):**
- Changed model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` → `meta-llama/Meta-Llama-3-8B`
- Added `gradient_checkpointing_enable()` for memory efficiency
- Added `torch_dtype="auto"` for optimal precision
- Reduced debug output frequency (1% instead of 5%)

**Cell 16 (Training Function):**
- Epochs: 3 → **10**
- Batch size: 4 → **1** (with 8× gradient accumulation = effective 8)
- Learning rate: default → **2e-5** (standard for large models)
- Warmup steps: 100 → **500**
- Eval steps: 200 → **1000** (less frequent for efficiency)
- Added `gradient_checkpointing=True`
- Added `save_total_limit=3` (keep only 3 checkpoints)
- Comprehensive training info printout

**Cell 18 (Experiment Function):**
- Removed 500-situation limit: Now uses ALL 10,000 situations
- Updated printouts to show scale
- Added `training_epochs` to results JSON

### Hardware Requirements

- **GPU**: A100 (40GB) recommended, minimum V100 (32GB)
- **RAM**: High-RAM runtime on Colab
- **Training time**: 4-6 hours (vs 5 minutes before)

### Expected Results

If the task IS learnable with more capacity:
- Accuracy > 0.6 (clearly better than random 0.5)
- Risk-averse preference > 0.6 (consistent preference)
- Score distributions separate (green vs red bars)
- Training loss continues decreasing over 10 epochs

If results are STILL 0.5 accuracy:
- Task may require chain-of-thought reasoning
- May need explicit CARA utility features
- May need even larger models (70B+)
- Or task is fundamentally not learnable from text alone

### Technical Rationale

This scale-up tests the hypothesis that **the task is learnable but requires more capacity and data**.

The 71× increase in total training should be sufficient for Llama-3-8B to:
1. Learn to parse probability statements
2. Approximate expected values from text
3. Discover the CARA utility pattern across 10K examples
4. Generalize to unseen test situations

If this fails, we'll have strong evidence the task requires a different approach.

## [2.5.1] - CRITICAL FIX: Data Corruption in Labels - 2025-11-04

### Major Bug Fix: Corrupted Training Labels

**🚨 ROOT CAUSE FOUND: The model was being trained on corrupted data from the start!**

### The Problem

The CSV data had a critical formatting issue:
- `correct_label` and `incorrect_label` contained **negative indices** (e.g., "-1", "-2")
- These indices **did not appear anywhere in the prompt text**
- Prompts showed options like "(1)", "(2)", "(3)", "(4)" or "a", "b", "c"
- But training examples said "Chosen option: -1" (which is never mentioned in context!)

**Example of the corruption:**
```
Prompt text lists: (1), (2), (3), (4)
correct_label: -1  ← Not in prompt!
incorrect_label: -2  ← Not in prompt!

Training input: "... options (1), (2), (3), (4) ... Chosen option: -1"
Model asked to learn: Score "-1" as 1.0, score "-2" as 0.0
But "-1" has ZERO relationship to the actual risk-averse option!
```

### Why Training Appeared to Work But Didn't

1. **Training loss decreased** ✓ - Model learned to distinguish strings "-1" vs "-2"
2. **But scores never separated** ✗ - Because:
   - The pattern "-1"=high, "-2"=low doesn't generalize to new test data
   - The negative indices are arbitrary and not semantically meaningful
   - Different situations use different label formats (numbers vs letters)
3. **Result**: 4 experiments, all with 0.5 accuracy and 0.0 preference

This explains ALL the failures - we never trained the model on the actual task!

### The Solution

Fixed `RiskAversionDataLoader` to map negative indices to actual option labels:

```python
# Map negative indices to actual labels from labels_vector
# -1 → labels_vector[0] = "(1)"
# -2 → labels_vector[1] = "(2)"
if correct_label.startswith('-'):
    index = abs(int(correct_label)) - 1
    labels_list = ast.literal_eval(labels_vector_str)
    correct_label = labels_list[index]  # Now matches prompt format!
```

### Changes

**Created `fix_csv_labels.py` script:**
- Standalone Python script to permanently fix the CSV file
- Maps negative indices to actual labels from `labels_vector` column
- Fixed 5,040 labels total (2,520 correct_label + 2,520 incorrect_label)
- Verification confirmed all negative labels successfully replaced
- Fixed CSV replaces the original file (no separate filename needed)

**Cell 6 (RiskAversionDataLoader):**
- Clean, simple data loader with no runtime fixing logic
- Uses original filename: `strict_disagreements_10k_with_prompts_and_bad_formats.csv`
- Data is now pre-cleaned, so no additional checks needed

**Example transformation:**
```
Before:
  prompt: "... (1), (2), (3), (4) ..."
  correct_label: "-1"
  incorrect_label: "-2"

After:
  prompt: "... (1), (2), (3), (4) ..."
  correct_label: "(1)"  ✓ Matches prompt!
  incorrect_label: "(2)"  ✓ Matches prompt!
```

### Expected Results

Now that the model is training on **correct, semantically meaningful labels**:
- Model should learn to associate specific options with risk-averse behavior
- Score distributions should finally separate
- Accuracy should exceed 0.5
- Risk-averse preference rate should exceed 0.0

This was the missing piece all along - we can't learn patterns from corrupted data!

## [2.5.0] - Switch to Pure Single-Input Training - 2025-11-04

### Major Change: Abandon Mixed Training Entirely

**After 3 failed experiments with mixed training, switching to pure single-input classification.**

### Problem Diagnosis

Mixed training (pairwise + single-input) **completely failed** across all variations:
1. **Original mixed training** (v2.4.0): Accuracy 0.5, preference 0.0, scores at ~0.003
2. **After removing L2** (v2.4.1): Accuracy 0.5, preference 0.0, scores at ~0.007
3. **Consistent pattern**: Training loss decreased, but scores never differentiated

**Root cause**: Pairwise and single-input training created **conflicting gradients**:
- Pairwise task: "Score option A higher than option B within same context" (relative scoring)
- Single-input task: "Score this option as 1.0 or 0.0" (absolute scoring)
- Same model weights receiving contradictory optimization signals
- Gradients likely canceling each other out → no learning

### Solution: Pure Single-Input Classification

**New approach - radical simplification**:
- Train **ONLY** with single-input binary classification
- Each example labeled: risk-averse=1.0, risk-neutral=0.0
- Simple BCE loss: `BCEWithLogitsLoss(score, label)`
- No pairwise comparisons, no margin loss, no mixed modes
- Training matches evaluation (both single-input)

### Changes

**Dataset (Cell 8):**
- Created simple `SingleInputDataset` class
- Generates 2 examples per situation (risk-averse + risk-neutral)
- Returns standard format: `{input_ids, attention_mask, labels}`
- Removed all mixed training complexity (MixedTrainingDataset, ModeGroupedBatchSampler, MixedDataCollator)

**Model (Cell 10):**
- Simplified to single forward pass only
- Removed all pairwise ranking logic
- Pure BCE loss on individual examples
- Increased debug print frequency (5% of batches)

**Training (Cell 16):**
- Uses standard HuggingFace Trainer (no custom batch sampler!)
- Increased batch size: 1 → 4 (no memory issues with simpler approach)
- Removed gradient accumulation (not needed with batch_size=4)
- Simple collate function for labels

**Documentation (Cell 9):**
- Completely rewritten to explain pure single-input approach
- Removed all pairwise/mixed training documentation
- Clear explanation of why mixed training failed

### Expected Results

If this works, we should see:
- **Score separation**: Risk-averse scores high (~0.7-1.0), risk-neutral low (~0.0-0.3)
- **Accuracy > 0.5**: Better than random chance
- **Risk-averse preference > 0.5**: Model prefers risk-averse options
- **Clear histogram separation**: Green and red bars not overlapping
- **Green zone scatter points**: Above diagonal line

If this **still** doesn't work, it suggests:
- The task itself is too hard for TinyLlama 1.1B
- The prompt format doesn't provide enough signal
- Need larger model or different approach entirely

### Technical Rationale

This is the simplest possible approach that matches training and evaluation:
1. **No mode confusion**: Only one type of example
2. **No gradient conflicts**: Single loss function
3. **Direct supervision**: Clear labels (1.0 vs 0.0)
4. **Standard architecture**: Works like any classification task

If a model can't learn this simple task, it can't learn the complex mixed version either.

### Hotfix: Device Mismatch in Forward Pass (Comprehensive Fix)
- **Error**: `Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA__index_select)`
- **Root Cause**: Multiple issues:
  1. Tensors from collate function were on CPU
  2. Manual validation step was passing CPU tensors to CUDA model
  3. HuggingFace Trainer doesn't always move all custom tensors to device
- **Fix**:
  1. **Cell 10**: Modified forward pass to explicitly move ALL input tensors (input_ids, attention_mask, labels) to model device before processing
  2. **Cell 16**: Removed manual validation step that was calling model with CPU tensors - let Trainer handle device placement
- **Technical Details**:
  - Get model device: `device = next(self.backbone.parameters()).device`
  - Move all inputs: `input_ids = input_ids.to(device)` (and same for attention_mask and labels)
  - Removed validation: No longer call `model(**sample_batch)` manually before training
- **Why This Works**: Ensures all tensors are on CUDA before backbone forward pass, and avoids manual calls that bypass Trainer's device management

## [2.4.1] - Experimental: Remove L2 Regularization - 2025-11-04

### Experimental Change
- **Removed L2 regularization from pairwise ranking loss**
- **Hypothesis**: L2 regularization was dominating the loss function and pushing all scores toward zero, preventing the model from learning meaningful score separations
- **Evidence from failed runs**:
  - Scores centered at ~0.003 (nearly zero)
  - Score distributions identical for both option types
  - Perfect diagonal in risk preference scatter plot (no differentiation)
  - Despite training loss decreasing, scores never moved away from initialization
- **Change**: Modified pairwise loss from `margin_loss + 0.1 × bradley_terry + 0.01 × L2` to `margin_loss + 0.1 × bradley_terry`
- **Reasoning**: L2 regularization penalty (0.01 × scores²) was fighting against margin loss (wants score_diff ≥ 1.0) and BCE loss (wants scores in 0-1 range). The conflict prevented gradient descent from moving scores away from zero.
- **Expected Result**:
  - Scores should spread out from zero
  - Risk-averse options should score higher (positive values)
  - Risk-neutral options should score lower (could go negative)
  - Score distributions should separate
- **Files Modified**:
  - Cell 9: Updated loss documentation table to show L2 removed, added warning about experimental change
  - Cell 10: Removed `score_regularization` calculation and removed it from `total_loss`, updated debug print statement label
- **Rollback Plan**: If this doesn't work, we can restore L2 with much smaller weight (0.0001) or try other theories

## [2.4.0] - Mixed Training to Fix Training-Evaluation Mismatch - 2025-11-04

### Major Changes
- **CRITICAL FIX**: Implemented mixed training combining pairwise ranking + single-input classification
- Replaced `PairwiseRiskAversionDataset` with `MixedTrainingDataset` that generates both pairwise and single-input examples
- Created `MixedDataCollator` to handle batches of different modes
- Updated model forward pass to detect and route between pairwise and single-input modes

### Problem Diagnosed
**Root cause of 0.5 accuracy and 0.0 risk-averse preference rate:**

Previous implementation trained exclusively with pairwise examples (comparing risk-averse vs risk-neutral options together), but evaluated with single-input examples (scoring options one at a time). This created a fundamental mismatch:

**Training (Pairwise Mode):**
- Model learned: "Score option A higher than option B"
- Learned **relative scoring**: which option is better
- Never learned **absolute scoring**: what scores actually mean

**Evaluation (Single-Input Mode):**
- Model scored each option independently
- No comparison context available
- All scores collapsed to ~0 (untrained default from L2 regularization)
- Result: 50% accuracy (random chance), 0% risk-averse preference

**Evidence from plots:**
- Score distributions: Both options identical, centered at 0
- Risk preference scatter: All points on diagonal (no differentiation)
- Score difference: +0.000 (perfect tie)
- Training loss decreased, but didn't translate to evaluation

### Solution: Mixed Training

**New approach trains model in BOTH modes:**

1. **Pairwise ranking** (33% of examples): Teaches relative scoring
   - Input: Both options from same situation
   - Loss: Hybrid (margin + Bradley-Terry + L2)
   - Learns: "Risk-averse should score higher than risk-neutral"

2. **Single-input classification** (67% of examples): Teaches absolute scoring
   - Input: Individual options with labels (risk-averse=1.0, risk-neutral=0.0)
   - Loss: Binary cross-entropy
   - Learns: "Risk-averse options should score ~1, risk-neutral ~0"

**Data expansion:**
- Each situation generates 3 training examples (1 pairwise + 2 single-input)
- Training data: 350 situations → 1,050 examples
- Model learns both relative preferences AND score semantics

### Added
- `MixedTrainingDataset` class that generates both pairwise and single-input examples
- `MixedDataCollator` for handling heterogeneous batches
- Mode detection in model forward pass (`mode` parameter)
- Debug logging for single-input training progress
- Comprehensive documentation of mixed training approach

### Changed
- Dataset: `PairwiseRiskAversionDataset` → `MixedTrainingDataset` for training
- Collator: `PairwiseDataCollator` → `MixedDataCollator` for training
- Training examples per situation: 1 → 3 (1 pairwise + 2 single-input)
- Model forward pass now detects mode and routes appropriately
- Single-input forward pass enhanced with training support and debug logging

### Fixed
- **Training-evaluation mismatch**: Model now trains on the same input format used during evaluation
- **Score collapse to zero**: Single-input training teaches absolute score meanings
- **No risk-averse preference**: Model now learns to differentiate individual options
- **Pairwise-only training limitation**: Combined approach leverages strengths of both methods

### Technical Rationale

The original pairwise-only approach is mathematically sound for learning rankings but fails when evaluation requires absolute scoring. Mixed training solves this by:

1. **Pairwise training** provides strong supervision for relative preferences (margin ensures clear separation)
2. **Single-input training** grounds these preferences in absolute scores (risk-averse=high, risk-neutral=low)
3. **Combined effect** enables the model to both compare options AND score them independently

This is similar to how human judgments work: we can compare options relatively ("A is better than B") and also evaluate them absolutely ("A is good, B is bad").

### Expected Improvements

After mixed training, we expect:
- Accuracy: 0.5 → 0.65-0.75 (better than random)
- Risk-averse preference rate: 0.0 → 0.60-0.75 (clear preference)
- Score difference: 0.0 → positive (risk-averse scores higher)
- Score distributions: Separated (risk-averse higher, risk-neutral lower)

### Hotfix: IndexError in DataLoader
- **Fixed**: Added `dataframe.reset_index(drop=True)` in `MixedTrainingDataset.__init__`
- **Reasoning**: After `train_test_split`, DataFrame indices are non-sequential (e.g., [1, 3, 4] instead of [0, 1, 2]). Using these indices with `iloc[]` causes IndexError. Resetting to 0-based sequential indices fixes the mismatch.
- **Changed**: Loop from `for idx, row in dataframe.iterrows()` to `for idx in range(len(self.data))`

### Hotfix: Mixed Mode Batches in DataLoader
- **Error**: `ValueError: Batch contains mixed modes: ['pairwise', 'single', 'single', 'pairwise']`
- **Root Cause**: HuggingFace Trainer's DataLoader randomly samples examples without respecting mode grouping. Since `MixedTrainingDataset` creates a flat list of interleaved pairwise and single-input examples, random sampling created batches mixing both modes. But `MixedDataCollator` and the model require homogeneous batches (all pairwise OR all single-input).
- **Solution**: Created custom `ModeGroupedBatchSampler` and `MixedTrainingTrainer`
  - `ModeGroupedBatchSampler`: Groups dataset indices by mode, creates homogeneous batches within each group, then shuffles batch order (not indices within batches)
  - `MixedTrainingTrainer`: Custom Trainer that overrides **both** `get_train_dataloader()` and `get_eval_dataloader()` to use the mode-grouped sampler
- **Technical Details**:
  - Sampler splits dataset into pairwise_indices and single_indices lists
  - Creates separate batches for each mode (respecting batch_size and drop_last)
  - Shuffles batch order for randomness while maintaining within-batch homogeneity
  - Trainer uses `batch_sampler` instead of standard sampler for DataLoader
  - **Critical**: Both training AND evaluation dataloaders need mode-grouped sampling (error occurred during eval steps)
- **Files Modified**:
  - Cell 8: Added `ModeGroupedBatchSampler` class with mode grouping logic
  - Cell 16: Added `MixedTrainingTrainer` class with both dataloader overrides, updated `train_reward_model()` to use it
- **Why This Works**: The sampler ensures DataLoader yields only homogeneous batches for both training and evaluation, eliminating the mixed-mode error while maintaining training randomness through batch-level shuffling.

## [2.3.0] - Colab T4 GPU Compatibility Fixes - 2025-11-04

### Added
- **Evaluation progress tracking**: Print progress every 25 test situations during model evaluation
  - **Reasoning**: Evaluation can take several minutes on 150+ test situations. Progress updates show current accuracy and risk-averse preference rate, helping users monitor that evaluation is progressing and see intermediate results.

### Major Changes
- **BREAKING**: Reduced batch size from 2 to 1 for T4 GPU memory compatibility
- **BREAKING**: Reduced sequence length from 256 to 128 tokens
- **BREAKING**: Removed Flash Attention 2 support (optional dependency causing errors)
- **BREAKING**: Removed `dtype=torch.float16` from model initialization (fp16 now handled by Trainer)

### Fixed
- **Flash Attention error**: Removed `attn_implementation="flash_attention_2"` which required flash_attn package installation and compilation (~3-5 min wait). Model now uses standard attention.
  - **Reasoning**: Flash Attention is a nice-to-have optimization (~10-15% speedup) but not worth the compilation time and potential compatibility issues. Prioritized simplicity and universal compatibility.

- **group_by_length error**: Removed `group_by_length=True` from TrainingArguments which was incompatible with pairwise dataset using custom key names.
  - **Reasoning**: Our `PairwiseRiskAversionDataset` uses `risk_averse_input_ids` and `risk_neutral_input_ids` instead of standard `input_ids`, causing Trainer's length inference to fail.

- **wandb API key prompt**: Added `report_to="none"` to disable Weights & Biases and other external experiment tracking.
  - **Reasoning**: Automatic wandb integration prompts for API keys. Users don't need external tracking for this educational notebook - local logs in `./logs` are sufficient.

- **Padding token error**: Added `model.backbone.config.pad_token_id = tokenizer.pad_token_id` after model initialization.
  - **Reasoning**: Setting `tokenizer.pad_token` alone isn't enough - the model config also needs `pad_token_id` for batch sizes > 1. Without this, model cannot handle padding in batched training.

- **fp16 gradient scaler error**: Removed `dtype=torch.float16` from model loading kwargs.
  - **Reasoning**: Double-applying fp16 (once at load time with `dtype`, once during training with `fp16=True`) causes gradient scaler conflicts. Model should load in fp32 and let Trainer apply fp16 mixed precision correctly.

- **CUDA OOM error**: Reduced `per_device_train_batch_size` from 2 to 1 and `max_length` from 256 to 128.
  - **Reasoning**: T4 GPUs have 15GB VRAM. TinyLlama 1.1B + batch_size=2 + seq_len=256 exceeded memory (~14.7GB used). New settings use ~5-6GB, well within limits. Trade-off: ~20% slower training (+3-5 min) for universal T4 compatibility.

### Changed
- Model now loads in fp32 and is converted to fp16 by Trainer for proper mixed precision
- Training arguments streamlined to 18 parameters (removed group_by_length)
- All tokenizer max_length calls updated from 256 to 128 (affects training, evaluation, and test inference)
- Documentation updated to specify T4 GPU optimization and memory constraints

### Performance Impact
- Training time: +3-5 minutes (~20% slower) due to smaller batch size
- Memory usage: Reduced by ~60-70% (14.7GB → 5-6GB)
- Model quality: Minimal impact (128 tokens sufficient for most prompts)

### Technical Rationale
The notebook now prioritizes **reliability and compatibility** over maximum performance. All changes address real errors encountered on Google Colab T4 GPUs. The memory optimizations ensure the experiment runs on the most common free Colab GPU tier while maintaining research validity.

## [2.2.0] - TinyLlama Model Switch - 2025-10-29

### Major Changes
- **BREAKING**: Switched from Qwen2.5-1.5B to TinyLlama-1.1B-Chat-v1.0
- Fixed deprecation warning: changed `torch_dtype` to `dtype`

### Technical Rationale
TinyLlama provides similar performance with slightly smaller size (1.1B vs 1.5B parameters) and is specifically designed for efficiency. Built on Llama 2 architecture with FlashAttention support, it offers better computational efficiency while maintaining the same capabilities for risk preference learning.

## [2.1.0] - Colab GPU Optimizations - 2025-10-29

### Major Changes
- **BREAKING**: Removed Apple Silicon MPS compatibility code
- **BREAKING**: Upgraded default model from Qwen2.5-0.5B to Qwen2.5-1.5B
- **BREAKING**: Increased sequence length from 128 to 256 tokens

### Added
- Flash Attention 2 support when available
- Fused AdamW optimizer (`adamw_torch_fused`) for better GPU utilization
- Automatic device mapping with `device_map="auto"`
- Memory pinning for faster GPU transfers
- Group by length for training efficiency
- Multiple data loading workers

### Changed
- Always use fp16 mixed precision (not conditional)
- Increased batch sizes: train=2, eval=4 (from 1,1)
- Reduced gradient accumulation steps to 2 (from 4)
- Enabled prediction_loss_only for efficiency
- Updated documentation to emphasize Colab-first approach

### Removed
- MPS device compatibility code and CPU fallbacks
- Conditional fp16 and device mapping logic
- Local execution optimizations

### Technical Rationale
The project now targets Google Colab exclusively, allowing for aggressive GPU optimizations that were previously disabled for local/MPS compatibility. The larger model and longer sequences provide better performance on risk preference learning with Colab's T4/V100 GPUs.

## [2.0.0] - Pairwise Ranking Implementation - 2025-10-29

### Major Changes
- **BREAKING**: Completely redesigned loss function from binary cross-entropy to pairwise ranking loss
- **BREAKING**: Replaced standard dataset with `PairwiseRiskAversionDataset` for direct option comparison
- **BREAKING**: Modified model architecture to support dual-mode operation (pairwise training + single evaluation)

### Added
- `PairwiseRiskAversionDataset` class for training with option pairs from same scenario
- `PairwiseDataCollator` for custom tensor batching of dual inputs
- Hybrid loss function combining margin ranking, sigmoid, and L2 regularization components
- Real-time training debugging with score distributions and preference rate monitoring
- Risk-averse preference rate as primary evaluation metric
- CPU fallback for MPS devices to handle pairwise training compatibility issues
- Enhanced evaluation metrics focusing on ranking rather than classification

### Changed
- Model forward pass now supports both pairwise and single input modes
- Training uses 500 situations (reduced from full dataset) for faster experimentation
- Evaluation emphasizes score differences rather than binary classification accuracy
- Memory optimizations: 0.5B model, 128 token sequences, batch size 1 with gradient accumulation
- Device management with conditional fp16 (CUDA only) and automatic MPS fallback

### Fixed
- Training loss decreasing without improving risk-averse preference (core motivation for v2.0)
- MPS device compatibility issues with placeholder storage during complex operations
- Gradient flow problems during training stagnation
- Various training configuration errors (save_steps alignment, dtype deprecation warnings)

### Technical Rationale
The v1.0 binary classification approach suffered from spurious correlation learning—models achieved high accuracy by memorizing position bias or option letters rather than understanding risk preferences. The pairwise ranking approach directly optimizes the target behavior: risk-averse choices scoring higher than risk-neutral alternatives within the same scenario context.

## [1.0.0] - Initial Implementation - 2025-10-29

### Added
- Initial experiment structure based on README analysis
- `RiskAversionDataLoader` for CSV data processing
- `RiskAversionDataset` with binary classification approach
- `RiskAverseRewardModel` wrapper around transformer backbone
- Basic training pipeline with Hugging Face Trainer
- Evaluation system with accuracy metrics
- Comprehensive plotting system with 4-panel visualizations
- Memory optimization strategies
- Cross-platform compatibility (CUDA/MPS/CPU)
- Test harness with `test_experiment.py`
- Output organization with timestamped results
- Google Colab notebook for cloud execution

### Technical Foundation
- Binary cross-entropy loss treating risk preference as classification
- Each scenario generates 2 training examples (risk-averse=1, risk-neutral=0)
- Standard transformer fine-tuning approach
- Qwen2.5-0.5B model for memory efficiency
- 80/20 train/validation split with best model selection

### Initial Data Processing
- CSV validation with required columns check
- Prompt text modification for output-only format
- Situation deduplication by grouping on situation_id
- Support for both local and Colab execution environments

### Legacy Components (Removed in v2.0)
- Binary classification dataset approach
- Single-mode model forward pass
- BCE loss optimization
- Classification-focused evaluation metrics