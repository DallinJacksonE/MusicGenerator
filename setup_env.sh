#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

ENV_NAME="venv"

echo "Creating virtual environment '${ENV_NAME}'..."
python3 -m venv ${ENV_NAME}

echo "Activating virtual environment..."
source ${ENV_NAME}/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch mapped for AMD ROCm
echo "Installing PyTorch for ROCm..."
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# Install the other standard dependencies required for the script
echo "Installing standard dependencies..."
pip install pretty_midi numpy kagglehub tqdm matplotlib

echo "-------------------------------------------------------"
echo "Setup complete! "
echo "To activate your environment and run the script, use:"
echo ""
echo "    source ${ENV_NAME}/bin/activate"
echo "    python music_generator.py"
echo "-------------------------------------------------------"
