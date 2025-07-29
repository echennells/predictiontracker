"""
Digital Ocean configuration settings
"""
import os
from pathlib import Path

# API Configuration
# First try environment variable, then load from main config
DIGITAL_OCEAN_API_KEY = os.getenv("DIGITAL_OCEAN_API_KEY", "")
if not DIGITAL_OCEAN_API_KEY:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from config.config import DIGITAL_OCEAN_API_KEY as CONFIG_DO_KEY
        DIGITAL_OCEAN_API_KEY = CONFIG_DO_KEY
    except:
        pass

DIGITAL_OCEAN_API_BASE = "https://api.digitalocean.com/v2"

# Droplet Configuration
DEFAULT_DROPLET_SIZE = "g-2vcpu-8gb"  # GPU droplet
DEFAULT_REGION = "nyc1"  # New York datacenter
DEFAULT_IMAGE = "ubuntu-22-04-x64"
DEFAULT_DROPLET_TAGS = ["xtotext", "transcription", "gpu"]

# GPU Droplet Sizes (as of 2024)
GPU_DROPLET_SIZES = {
    "gpu-h100x1-80gb": {"vcpus": 20, "memory": 240, "disk": 3000, "gpu": "H100 80GB"},
    "gpu-h100x8-640gb": {"vcpus": 160, "memory": 1920, "disk": 24000, "gpu": "8x H100 80GB"},
    "g-2vcpu-8gb": {"vcpus": 2, "memory": 8, "disk": 50, "gpu": "None"},  # For testing
}

# SSH Configuration
SSH_KEY_NAME = "xtotext-key"
SSH_USERNAME = "root"
SSH_PORT = 22
SSH_TIMEOUT = 30
CONNECTION_RETRIES = 5
RETRY_DELAY = 5

# Paths
LOCAL_SSH_KEY_PATH = Path.home() / ".ssh" / "id_rsa"
REMOTE_WORKSPACE = "/workspace"
REMOTE_AUDIO_DIR = f"{REMOTE_WORKSPACE}/audio"
REMOTE_OUTPUT_DIR = f"{REMOTE_WORKSPACE}/transcripts"
REMOTE_CODE_DIR = f"{REMOTE_WORKSPACE}/xtotext"

# Instance Setup Script
SETUP_SCRIPT = """#!/bin/bash
set -e

echo "Starting Digital Ocean droplet setup..."

# Update system
apt-get update
apt-get install -y python3-pip python3-venv

# Create working directories
mkdir -p {REMOTE_WORKSPACE}
mkdir -p {REMOTE_CODE_DIR}

echo "Setup complete!"

# Create a flag file to indicate setup is done
touch /workspace/.setup_complete
echo "Setup completed at $(date)" > /workspace/.setup_complete
""".format(
    REMOTE_WORKSPACE=REMOTE_WORKSPACE,
    REMOTE_AUDIO_DIR=REMOTE_AUDIO_DIR,
    REMOTE_OUTPUT_DIR=REMOTE_OUTPUT_DIR,
    REMOTE_CODE_DIR=REMOTE_CODE_DIR
)

# Job Configuration
MAX_CONCURRENT_JOBS = 5
JOB_TIMEOUT = 3600  # 1 hour default
JOB_CHECK_INTERVAL = 10  # seconds

# Control Tower Configuration
CONTROL_TOWER_PORT = 8080
CONTROL_TOWER_HOST = "0.0.0.0"
API_SECRET_KEY = os.getenv("CONTROL_TOWER_SECRET", "change-me-in-production")