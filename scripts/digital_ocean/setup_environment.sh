#!/bin/bash
# Setup script for Digital Ocean droplet
set -e

echo "Setting up Python environment on Digital Ocean..."

cd /workspace

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required dependencies
echo "Installing dependencies..."
pip install openai
pip install anthropic
pip install pydantic
pip install typing-extensions
pip install ijson

echo "Environment setup complete!"