# Risk-Averse Reward Model

## Overview

A misaligned future AI system might pursue catastrophic high-variance plans (up to and including attempts at world takeover) only because such plans have positive *expected* value. One way to make this less likely is to instill **risk-aversion** where an agent that places extra weight on bad outcomes will pass on high-variance gambles even when they look attractive in expectation, and will instead prefer cooperative, lower-variance options. The open empirical question is whether such a disposition, trained on cheap low-stakes decision problems, **generalizes** to the high-stakes, out-of-distribution settings that actually matter.

This experiment trains reward models to prefer choices consistent with a CARA utility (u(w) = 1 − e^(−0.01w)) over two failure modes — risk-neutral expected-value reasoning and *over*-risk-aversion (α = 0.10) — using Chain-of-Thought pairwise preference data on small monetary gambles. We then test how the learned preference transfers to scenarios it never saw during training: high-stakes payoffs, astronomical-stakes deployment scenarios, and cooperate-vs-steal social dilemmas. Each base model (Qwen3-1.7B / 8B / 14B, Llama-3.1-8B-Instruct, Gemma-3-12b-it) is also benchmarked on Reward-Bench 2 to confirm the risk-aversion training does not destroy general reward-model quality.

This repository contains the full pipeline: the training script (`rft_pipeline.py`), the per-dataset evaluators it invokes, the CoT training and eval CSVs (under the `eval/` submodule), and a resumable run-and-summary layout that produces per-seed metrics plus mean ± SD aggregates. A single command runs the LR sweep, picks the best LR by validation accuracy, retrains across seeds 1/2/3, and writes all heldout + RB2 results to a single `final_summary.json`.

## Architecture

- **Backbone**: HuggingFace `AutoModel` (Qwen3-1.7B / 8B / 14B, Llama-3.1-8B-Instruct, or Gemma-3-12b-it), frozen.
- **LoRA** (`peft`, r=32, α=64) over the standard q/k/v/o + gate/up/down projections.
- **Reward head**: `nn.Linear(hidden_size, 1)` in fp32, applied to the last non-padding token's hidden state.
- **Loss**: `−log σ(r_chosen − r_rejected)` (Bradley–Terry).
- Mixed precision: backbone fp16/bf16, head fp32 (essential for stability).

## Setup

```bash
git clone --recurse-submodules <this-repo>
cd risk-averse-reward-model
source setup.sh        # creates .venv, installs requirements.txt, pulls eval/ submodule
```

The submodule (`eval/risk-averse-ai-eval/`) provides the training and eval CSVs plus `evaluate_reward_model.py`. Base-model weights are pulled from HF on first run; gated models (Llama, Gemma) require `HF_TOKEN`.

## Run

The full pipeline runs in two phases — an LR sweep on validation, then heldout + Reward-Bench 2 across seeds 1/2/3:

```bash
python rft_pipeline.py --base_model Qwen/Qwen3-8B --middle_lr 5e-4 --epochs 5
```

Common variants:

```bash
# Locked-config single run (no LR ladder, one seed)
python rft_pipeline.py --base_model Qwen/Qwen3-14B \
    --middle_lr 5e-4 --epochs 7 --single_lr \
    --seeds_validation 1 --seeds_heldout 1

# Untrained-baseline measurement (random-init reward head)
python rft_pipeline.py --base_model Qwen/Qwen3-8B \
    --no_train --single_lr --seeds_validation 1 --seeds_heldout 1

# Skip phases or constrain compute
python rft_pipeline.py --skip_heldout --skip_reward_bench_2
python rft_pipeline.py --dry_run               # validate paths without GPU
```

Recommended GPU: H100/H200 80GB. Memory knobs: `--max_length 768`, `--batch_size 1`, `--no_grad_ckpt`. See `python rft_pipeline.py --help` for the full flag set.

## Outputs

Each run writes under `outputs/<output_subdir>/`:

```
validation/lr_<lr>_seed_<S>/
    checkpoint/                              # LoRA adapter + reward_head.pt
    eval_rm_reward_model_validation.json     # per-seed metrics
heldout/lr_<lr>_seed_<S>/
    eval_rm_reward_model_high_stakes_test.json
    eval_rm_reward_model_astronomical_stakes_deployment.json
    eval_rm_reward_model_steals_test.json
    eval_reward_bench_2.json                 # 5 standard subsets + Ties
final_summary.json                           # mean ± SD across seeds
status.json                                  # resumable run ledger
```

Runs are resumable — `complete.json` per run dir and `eval_rm_*.json` per dataset act as sentinels.

## Data

Training (`eval/risk-averse-ai-eval/data/2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv`): 1000 situations, each with a full Chain-of-Thought response for both the CARA-correct and an incorrect (risk-neutral or over-risk-averse) answer. Each row is one training pair.

Eval CSVs (200–928 pairs each):
- `reward_model_validation` — in-distribution, low stakes
- `reward_model_high_stakes_test` — OOD, scaled-up payoffs
- `reward_model_astronomical_stakes_deployment` — OOD, deployment-framed
- `reward_model_steals_test` — OOD, cooperate-vs-steal framing

For the validation/heldout pairs, the **chosen** option is the cooperative / CARA-correct answer; **rejected** is the risk-neutral or risk-seeking alternative.

## Files

```
rft_pipeline.py                              # entry point
evaluate_reward_bench_2.py                   # invoked for the RB2 phase
requirements.txt                             # Python deps
setup.sh                                     # bootstrap
eval/risk-averse-ai-eval/                    # submodule: data + per-dataset eval
    evaluate_reward_model.py                 # invoked for per-dataset RM eval
    risk_averse_prompts.py
    cot_csv_utils.py
    data/                                    # training + eval CSVs
```
