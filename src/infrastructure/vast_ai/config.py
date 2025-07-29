"""
Vast.ai configuration settings
"""
import os
from pathlib import Path

# API Configuration
VAST_API_KEY = os.getenv("VAST_API_KEY", "")
VAST_API_BASE = "https://vast.ai/api/v0"

# Instance Configuration
DEFAULT_GPU_TYPE = "RTX 3080"
DEFAULT_GPU_COUNT = 1
DEFAULT_MIN_GPU_RAM = 10  # GB
DEFAULT_MIN_RAM = 16  # GB
DEFAULT_MIN_DISK = 50  # GB
DEFAULT_MAX_PRICE = 1.00  # USD per hour (increased to find more options globally)

# SSH Configuration
SSH_KEY_PATH = Path.home() / ".ssh" / "id_rsa"
SSH_USERNAME = "root"
SSH_PORT = 22
SSH_TIMEOUT = 30  # seconds
CONNECTION_RETRIES = 5
RETRY_DELAY = 5  # seconds

# Whisper Configuration  
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# Instance Setup Script
SETUP_SCRIPT = """#!/bin/bash
set -e

echo "Starting Vast.ai instance setup..."

# Update system
apt-get update
apt-get install -y python3-pip ffmpeg git

# Install CUDA support
apt-get install -y nvidia-cuda-toolkit

# Install Whisper and dependencies
pip install --upgrade pip
pip install openai-whisper
pip install faster-whisper  # GPU-optimized version

# Create working directory
mkdir -p /workspace
cd /workspace

echo "Setup complete!"
"""

# Paths
REMOTE_WORKSPACE = "/workspace"
REMOTE_AUDIO_DIR = f"{REMOTE_WORKSPACE}/audio"
REMOTE_OUTPUT_DIR = f"{REMOTE_WORKSPACE}/transcripts"