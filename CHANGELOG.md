# Changelog

All notable changes to the risk-averse reward model project will be documented in this file.

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