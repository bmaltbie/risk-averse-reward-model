#!/bin/bash
# Setup Script
# Usage: source setup.sh

set -e  # Exit on error

echo "=== Setting up environment ==="

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Virtual environment: .venv (activated)"
echo ""
