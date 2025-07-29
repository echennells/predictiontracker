# Video to Text Transcription Tool

A modular Python tool for downloading videos from YouTube and X (Twitter) and transcribing them using OpenAI's Whisper API.

## Features

- Download videos from YouTube and X (Twitter)
- Extract audio from video files
- Transcribe audio/video using OpenAI Whisper API
- Modular architecture for easy extension
- Cost estimation for API usage
- Timestamped transcription segments

## Setup

1. Install system dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### Download and transcribe a video:
```bash
# YouTube
python main.py https://youtube.com/watch?v=VIDEO_ID

# X (Twitter)
python main.py https://x.com/user/status/123456789
```

### Transcribe an existing file:
```bash
python main.py --transcribe-only video.mp4
```

### Download only (no transcription):
```bash
python main.py --download-only https://youtube.com/watch?v=VIDEO_ID
```

### Specify output file:
```bash
python main.py --output transcript.txt video.mp4
```

### Exclude timestamp segments:
```bash
python main.py --no-segments video.mp4
```

## Project Structure

```
.
├── main.py              # Main CLI interface
├── config.py            # Configuration settings
├── transcriber.py       # Audio transcription module
├── downloaders/         # Video downloader modules
│   ├── __init__.py
│   ├── base.py         # Base downloader class
│   ├── youtube.py      # YouTube downloader
│   └── x_downloader.py # X (Twitter) downloader
├── downloads/          # Downloaded videos directory
├── temp/              # Temporary files directory
└── requirements.txt   # Python dependencies
```

## API Costs

OpenAI Whisper API costs $0.006 per minute of audio. The tool displays estimated costs after each transcription.

## Extending

To add support for a new platform:

1. Create a new downloader class in `downloaders/` that inherits from `BaseDownloader`
2. Implement `validate_url()` and `get_platform_name()` methods
3. Import it in `downloaders/__init__.py`
4. Add it to the downloaders list in `main.py`