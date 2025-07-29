# Vast.ai GPU Transcription Module

This module provides GPU-accelerated transcription using Vast.ai cloud GPU instances.

## Setup

1. **Get Vast.ai API Key**
   - Sign up at https://vast.ai
   - Go to Account settings: https://vast.ai/console/account
   - Create an API key
   - Set environment variable: `export VAST_API_KEY="your-api-key"`

2. **Set up SSH Key**
   - Ensure you have an SSH key pair: `~/.ssh/id_rsa` and `~/.ssh/id_rsa.pub`
   - If not, create one: `ssh-keygen -t rsa -b 4096`

3. **Install Dependencies**
   ```bash
   pip install requests
   ```

## Usage

### Basic Transcription

```python
from vast_ai import TranscriptionRunner

# Initialize runner
runner = TranscriptionRunner()

# Set up GPU instance (RTX 3080)
instance = runner.setup_instance(
    gpu_type="RTX 3080",
    max_price=0.50  # Max $0.50/hour
)

# Transcribe audio file
result = runner.transcribe_audio(
    audio_path=Path("podcast.mp3"),
    model="large-v3",
    language="en"
)

# Clean up (optionally destroy instance)
runner.cleanup(destroy_instance=True)
```

### Command Line Usage

**Transcribe with GPU:**
```bash
python transcribe_with_gpu.py audio.mp3 --model large-v3 --destroy-after
```

**Manage instances:**
```bash
# Check status
python vast_manage.py status

# Create instance
python vast_manage.py create --gpu-type "RTX 3080" --max-price 0.50

# Destroy instance
python vast_manage.py destroy

# List all instances
python vast_manage.py list

# Estimate cost
python vast_manage.py estimate audio.mp3
```

### Integration with Main Transcriber

```python
from gpu_transcriber import GPUTranscriber

# Create GPU transcriber
transcriber = GPUTranscriber(
    gpu_type="RTX 3080",
    max_price=0.50
)

# Transcribe video
result = transcriber.transcribe_video(
    "video.mp4",
    model="large-v3"
)

# Get cost
cost = transcriber.get_instance_cost()
print(f"Total cost: ${cost:.4f}")

# Clean up
transcriber.cleanup(destroy_instance=True)
```

## GPU Options

The module automatically finds the cheapest instance globally. Common GPU types and typical global prices:
- **RTX 3080** (10GB): ~$0.15-0.50/hour - Good for most transcriptions
- **RTX 3090** (24GB): ~$0.20-0.70/hour - For very long audio files  
- **RTX 4090** (24GB): ~$0.30-1.00/hour - Fastest option
- **A100** (40GB): ~$0.50-2.00/hour - Professional grade

Note: Prices vary significantly by region. The module searches globally for the absolute cheapest option.

## Cost Optimization

1. **Instance Management**
   - Keep instance running for batch jobs
   - Destroy immediately after single transcriptions
   - Check instance status regularly

2. **Model Selection**
   - `base`: Fastest, lowest cost
   - `small`: Good balance
   - `medium`: Better accuracy
   - `large-v3`: Best accuracy, slower

3. **Batch Processing**
   ```python
   # Transcribe multiple files on same instance
   results = runner.transcribe_batch(
       audio_files=[Path("ep1.mp3"), Path("ep2.mp3")],
       output_dir=Path("transcripts/"),
       model="large-v3"
   )
   ```

## Troubleshooting

1. **SSH Connection Issues**
   - Ensure SSH key exists: `ls ~/.ssh/id_rsa*`
   - Check key permissions: `chmod 600 ~/.ssh/id_rsa`

2. **Instance Creation Fails**
   - Check available balance on Vast.ai
   - Try different GPU type or higher max price
   - Check for regional availability

3. **Transcription Errors**
   - Verify audio file is valid
   - Check instance has enough disk space
   - Monitor GPU memory usage

## Advanced Configuration

Edit `vast_ai/config.py` to customize:
- Default GPU specifications
- SSH settings
- Whisper model defaults
- Remote paths
- Setup scripts