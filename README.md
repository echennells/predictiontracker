Financial Prediction Extraction System

A sophisticated multi-stage system for automatically extracting and tracking financial predictions from podcast transcripts. This tool uses advanced LLM processing to identify, extract, and analyze price predictions with high accuracy and comprehensive metadata.

## Overview

processes podcast audio files to:
- Transcribe audio using GPU-accelerated Whisper models
- Extract financial predictions using a three-stage LLM pipeline
- Track predictions with timestamps, confidence levels, and timeframes
- Generate structured data for analysis and backtesting

## Key Features

### 🎯 Multi-Stage Extraction Pipeline
- **Stage 1**: Fast scanning to locate potential predictions in transcripts
- **Stage 2**: Focused extraction with concurrent processing for speed
- **Stage 3**: Advanced timeframe parsing and validation

### ⚡ High Performance
- GPU-accelerated transcription via Vast.ai integration
- Concurrent API processing (3x faster than sequential)
- Smart batching to optimize token usage
- Automatic rate limit handling

### 📊 Comprehensive Data Extraction
- Asset identification (stocks, cryptocurrencies, ETFs)
- Price targets and percentage changes
- Timeframe parsing (specific dates, relative timeframes)
- Speaker attribution and confidence levels
- Exact timestamps with YouTube links

### 🔧 Flexible Architecture
- Modular LLM client system (GPT-4, Claude, etc.)
- Configurable models per stage
- Cost optimization with model selection
- Dry-run mode for testing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/xtotext.git
cd xtotext

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy the example configuration:
```bash
cp config/config.example.py config/config.py
```

2. Set your API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export DIGITAL_OCEAN_API_KEY="your-do-key"
export VAST_API_KEY="your-vast-key"
```

3. Configure model selection (optional):
```bash
export STAGE1_MODEL="gpt-4o"        # Fast scanning
export STAGE2_MODEL="gpt-4o"        # Extraction
export STAGE3_MODEL="gpt-4-turbo"   # Refinement
```

## Usage

### Basic Usage

```bash
# Process a podcast channel
python main.py
```

The system will:
1. Download new episodes from the configured channel
2. Transcribe audio using GPU instances
3. Extract predictions using the LLM pipeline
4. Save results to `data/episodes/[channel]/prediction_data/`

### Output Format

Predictions are saved as JSON with the following structure:

```json
{
  "text": "BTC to $150,000",
  "asset": "BTC",
  "value": 150000.0,
  "confidence": "high",
  "timestamp": "1:23:45",
  "timeframe": "end of 2025",
  "context": "I think Bitcoin will reach $150k by end of next year...",
  "episode": "Episode Title",
  "episode_date": "2024-01-15",
  "reasoning": "Based on halving cycle and institutional adoption"
}
```

## Architecture

### Processing Pipeline

```
Audio Files → Transcription (Vast.ai GPU) → Stage 1 (Scanning) → Stage 2 (Extraction) → Stage 3 (Refinement) → Structured Data
```

### Key Components

- **Downloaders**: YouTube channel monitoring and downloading
- **Infrastructure**: Vast.ai GPU management, Digital Ocean processing
- **LLM Clients**: Modular system supporting multiple models
- **Prediction Tracker**: Core extraction and processing logic
- **Storage**: JSON-based prediction database

## Advanced Features

### Concurrent Processing
Enable faster extraction with parallel API calls:
```bash
export ENABLE_CONCURRENT_PROCESSING=true
export CONCURRENT_WORKERS=3
```

### Dry Run Mode
Test Stage 3 without making API calls:
```bash
export STAGE3_DRY_RUN=true
```

### Custom Prompts
Configure extraction behavior in `llm_extractor_two_stage.py`:
- Asset name mappings
- Confidence thresholds
- Context window sizes

## Performance

Typical processing times:
- Transcription: 2-3 minutes per hour of audio
- Extraction: 3-4 minutes per episode
- Total: ~45 minutes for 12 episodes with concurrent processing

## Cost Optimization

Approximate costs per episode:
- Stage 1: $0.05-0.10
- Stage 2: $0.10-0.20
- Stage 3: $0.01-0.02
- Total: $0.16-0.32 per episode

## Troubleshooting

### API Rate Limits
The system handles rate limits automatically with:
- Exponential backoff
- Request queuing
- Automatic retries

### Memory Issues
For large transcripts:
- Adjust `CHUNK_SIZE` in config
- Enable garbage collection
- Process episodes individually

### Accuracy Issues
To improve extraction accuracy:
- Upgrade Stage 1/2 models
- Adjust asset name mappings
- Fine-tune confidence thresholds

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Disclaimer

This tool is for research and analysis purposes only. Always verify predictions and perform your own due diligence before making financial decisions.
