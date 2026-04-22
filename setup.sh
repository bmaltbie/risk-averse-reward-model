#!/bin/bash
# Setup Script — bootstraps rft_pipeline.py on a fresh runpod (or any clean host).
#
# Pulls everything that can be pulled:
#   1. The eval submodule (training + eval CSVs) from GitHub
#   2. Python dependencies from PyPI
# Model weights for Qwen/Qwen3-* are pulled lazily by HuggingFace at runtime.
#
# Usage:
#   source setup.sh    # activates .venv in the current shell (recommended)
#   bash setup.sh      # runs in a subshell; .venv activation is lost after exit

set -e

# ------------------------------------------------------------
# 0. Optional: HuggingFace access token
# ------------------------------------------------------------
# Required only for gated models (e.g., meta-llama/*). Qwen3 is public and
# needs no token. To enable, edit the line below and uncomment.
# export HF_TOKEN="hf_your_token_here"

echo "=== Setting up environment ==="

# ------------------------------------------------------------
# 1. Submodule: training + eval CSVs (pull, don't copy locally)
# ------------------------------------------------------------
SUBMODULE_PATH="eval/risk-averse-ai-eval"
SUBMODULE_URL="https://github.com/elliottthornley/risk-averse-ai-eval"

if [ -f .gitmodules ] && [ -d .git ]; then
    # Standard case: parent repo was cloned with submodule registered.
    echo "Updating git submodule ${SUBMODULE_PATH}..."
    git submodule update --init --recursive
elif [ -d "${SUBMODULE_PATH}/.git" ] || [ -f "${SUBMODULE_PATH}/.git" ]; then
    # Submodule dir already present (previously cloned standalone); pull latest.
    echo "Submodule present at ${SUBMODULE_PATH}; pulling latest..."
    git -C "${SUBMODULE_PATH}" pull --ff-only
else
    # Tarball / minimal-upload case: clone the submodule directly.
    echo "Cloning ${SUBMODULE_URL} into ${SUBMODULE_PATH}..."
    mkdir -p "$(dirname "${SUBMODULE_PATH}")"
    git clone "${SUBMODULE_URL}" "${SUBMODULE_PATH}"
fi

# Sanity-check the CSVs rft_pipeline.py reads at runtime.
REQUIRED_CSVS=(
    "${SUBMODULE_PATH}/data/2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv"
    "${SUBMODULE_PATH}/data/2026_03_22_reward_model_val_set_400_Rebels_clean.csv"
    "${SUBMODULE_PATH}/data/2026_03_22_high_stakes_test_set_746_Rebels_CoTs_for_evaluating_reward_model_from_Sonnet.csv"
    "${SUBMODULE_PATH}/data/2026_03_22_astronomical_stakes_deployment_set_707_Rebels_CoTs_for_evaluating_reward_model_from_Sonnet.csv"
    "${SUBMODULE_PATH}/data/2026_03_22_test_set_928_Steals_CoTs_for_evaluating_reward_model_from_Sonnet.csv"
)
missing_csvs=0
for f in "${REQUIRED_CSVS[@]}"; do
    if [ ! -f "$f" ]; then
        echo "WARNING: missing CSV $f"
        missing_csvs=$((missing_csvs + 1))
    fi
done
if [ "$missing_csvs" -gt 0 ]; then
    echo "WARNING: ${missing_csvs} required CSVs missing; rft_pipeline.py will fail at runtime."
fi

# ------------------------------------------------------------
# 2. Python virtual environment + dependencies
# ------------------------------------------------------------
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# ------------------------------------------------------------
# 3. Summary
# ------------------------------------------------------------
submodule_sha="$(git -C "${SUBMODULE_PATH}" rev-parse --short HEAD 2>/dev/null || echo '?')"
hf_token_status="not set (Qwen3 is public — fine)"
if [ -n "${HF_TOKEN:-}" ]; then hf_token_status="set"; fi

echo ""
echo "=== Setup complete ==="
echo "Virtual environment: .venv (activated)"
echo "Submodule:           ${SUBMODULE_PATH} @ ${submodule_sha}"
echo "CSVs present:        $((${#REQUIRED_CSVS[@]} - missing_csvs)) / ${#REQUIRED_CSVS[@]}"
echo "HF_TOKEN:            ${hf_token_status}"
echo ""
echo "Next: python rft_pipeline.py --dry_run"
echo ""
