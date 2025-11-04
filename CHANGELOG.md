# Changelog

All notable changes to the risk-averse reward model project will be documented in this file.

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