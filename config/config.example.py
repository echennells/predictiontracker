"""
Configuration module for video transcription system

Copy this file to config.py and update with your API keys:
    cp config/config.example.py config/config.py
"""
import os
from pathlib import Path

# API Configuration
# Get your API keys from:
# - OpenAI: https://platform.openai.com/api-keys
# - OpenRouter: https://openrouter.ai/keys
# - Digital Ocean: https://cloud.digitalocean.com/account/api/tokens
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-openrouter-api-key-here")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DIGITAL_OCEAN_API_KEY = os.getenv("DIGITAL_OCEAN_API_KEY", "your-digital-ocean-api-key-here")

# Directory Configuration
BASE_DIR = Path(__file__).parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
TEMP_DIR = BASE_DIR / "temp"

# Audio Configuration
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# Transcription Configuration
WHISPER_MODEL = "whisper-1"
TRANSCRIPTION_LANGUAGE = "en"
COST_PER_MINUTE = 0.006  # $0.006 per minute for Whisper API

# Video Download Configuration
VIDEO_FORMAT = "best[ext=mp4]/best"
DEFAULT_OUTPUT_PATTERN = "%(title)s.%(ext)s"

# LLM Configuration for Prediction Extraction
# Available models and their context windows:
# - "gpt-4-turbo" (128k tokens) - Best accuracy, most expensive
# - "gpt-4o" (128k tokens) - Good balance of speed/accuracy/cost
# - "gpt-4o-mini" (128k tokens) - Fastest and cheapest, lower accuracy
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")  # Default to GPT-4o for better cost/performance balance

# Stage-specific model configuration
# Stage 1: Initial scan - Find potential prediction areas (cheapest)
STAGE1_MODEL = os.getenv("STAGE1_MODEL", "gpt-4o")
# Stage 2: Focused extraction - Extract structured predictions (balanced)
STAGE2_MODEL = os.getenv("STAGE2_MODEL", "gpt-4o")  # Upgraded from gpt-4o-mini for better accuracy
# Stage 3: Detailed extraction - High-value deep analysis (most expensive)
STAGE3_MODEL = os.getenv("STAGE3_MODEL", "gpt-4-turbo")
# Dry run mode for Stage 3 - log what would be sent without making API calls
STAGE3_DRY_RUN = os.getenv("STAGE3_DRY_RUN", "false").lower() == "true"

LLM_TEMPERATURE = 0  # Zero temperature for maximum consistency in extraction
LLM_MAX_TOKENS = 2000

# Model-specific chunk sizes to avoid rate limits
# Your account has 30k TPM limit, so we need smaller chunks
# 28k tokens = ~112k chars, leaving 2k tokens for prompt/response
MODEL_CHUNK_SIZES = {
    "gpt-4-turbo": 112000,  # ~28k tokens, leaving room for prompt
    "gpt-4o": 112000,       # ~28k tokens, leaving room for prompt  
    "gpt-4o-mini": 112000,  # ~28k tokens, leaving room for prompt
}

# Get chunk size based on current model
CHUNK_SIZE = MODEL_CHUNK_SIZES.get(LLM_MODEL, 120000)
CHUNK_OVERLAP = 20000  # Increased overlap to avoid missing predictions at boundaries

# Concurrent Processing Configuration
# Enable concurrent API calls for Stage 2 prediction extraction
ENABLE_CONCURRENT_PROCESSING = os.getenv("ENABLE_CONCURRENT_PROCESSING", "true").lower() == "true"
# Number of concurrent workers for API calls (default 3, max recommended 5)
CONCURRENT_WORKERS = int(os.getenv("CONCURRENT_WORKERS", "3"))

# Ensure directories exist
DOWNLOADS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)