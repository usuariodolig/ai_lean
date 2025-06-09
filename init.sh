#!/bin/bash

# Print each command before executing it
set -x

# Exit on any error
set -e

# Install required Python packages
python -m pip install --upgrade pip
pip install --upgrade torch accelerate transformers openai matplotlib google-genai

# Configure git
git config --global user.name "usuariodolig"
git config --global user.email "usuariodolig@gmail.com"

# Change directory and run setup script
python download_model.py