"""
Transcription Runner for Vast.ai GPU Instances
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import time
from datetime import datetime

from .instance_manager import InstanceManager
from .ssh_connection import SSHConnection
from .config import (
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    REMOTE_WORKSPACE, REMOTE_AUDIO_DIR, REMOTE_OUTPUT_DIR
)


class TranscriptionRunner:
    """Runs Whisper transcription on Vast.ai GPU instances"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.instance_manager = InstanceManager(api_key)
        self.ssh_connection = None
    
    def setup_instance(
        self,
        gpu_type: str = "RTX 3080",
        max_price: float = 0.50
    ) -> Dict[str, Any]:
        """
        Set up a GPU instance for transcription
        
        Returns:
            Instance details
        """
        # Create or get existing instance
        instance = self.instance_manager.create_transcription_instance(
            gpu_type=gpu_type,
            max_price=max_price
        )
        
        # Establish SSH connection
        self.ssh_connection = SSHConnection(
            host=instance['ssh_host'],
            port=instance['ssh_port']
        )
        
        # Wait for SSH to be ready with a longer timeout
        print("Waiting for SSH to be ready (this can take 2-3 minutes)...")
        if not self.ssh_connection.wait_for_connection(timeout=600):
            raise RuntimeError("Failed to establish SSH connection")
        
        # Verify setup completed
        print("Verifying instance setup...")
        self._verify_setup()
        
        return instance
    
    def transcribe_audio(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None,
        model: str = WHISPER_MODEL,
        language: Optional[str] = None,
        use_faster_whisper: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using GPU instance
        
        Args:
            audio_path: Local audio file path
            output_dir: Local directory for output (default: same as audio)
            model: Whisper model to use
            language: Language code (None for auto-detect)
            use_faster_whisper: Use faster-whisper implementation
            
        Returns:
            Transcription results
        """
        if not self.ssh_connection:
            raise RuntimeError("No instance available. Run setup_instance() first.")
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Default output directory
        if output_dir is None:
            output_dir = audio_path.parent
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Remote paths
        remote_audio = f"{REMOTE_AUDIO_DIR}/{audio_path.name}"
        remote_output_base = f"{REMOTE_OUTPUT_DIR}/{audio_path.stem}"
        
        print(f"\\nTranscribing {audio_path.name} using {model} model...")
        start_time = time.time()
        
        try:
            # Create remote directories
            self.ssh_connection.execute_command(
                f"mkdir -p {REMOTE_AUDIO_DIR} {REMOTE_OUTPUT_DIR}"
            )
            
            # Upload audio file
            print("Uploading audio file...")
            if not self.ssh_connection.upload_file(audio_path, remote_audio):
                raise RuntimeError("Failed to upload audio file")
            
            # Run transcription
            if use_faster_whisper:
                transcript_path = self._run_faster_whisper(
                    remote_audio,
                    remote_output_base,
                    model,
                    language
                )
            else:
                transcript_path = self._run_whisper(
                    remote_audio,
                    remote_output_base,
                    model,
                    language
                )
            
            # Download results
            local_output = output_dir / f"{audio_path.stem}_transcript.json"
            print("Downloading transcript...")
            if not self.ssh_connection.download_file(transcript_path, local_output):
                raise RuntimeError("Failed to download transcript")
            
            # Load and add metadata
            with open(local_output, 'r') as f:
                transcript = json.load(f)
            
            # Add metadata
            duration = time.time() - start_time
            transcript['metadata'] = {
                'audio_file': audio_path.name,
                'model': model,
                'device': 'GPU',
                'transcription_time': duration,
                'timestamp': datetime.now().isoformat(),
                'instance_cost': self.instance_manager.estimate_cost(duration / 60)
            }
            
            # Save updated transcript
            with open(local_output, 'w') as f:
                json.dump(transcript, f, indent=2)
            
            print(f"\\nTranscription complete in {duration:.1f} seconds")
            print(f"Output saved to: {local_output}")
            print(f"Estimated cost: ${transcript['metadata']['instance_cost']:.4f}")
            
            return transcript
            
        finally:
            # Clean up remote files
            print("Cleaning up remote files...")
            self.ssh_connection.execute_command(
                f"rm -f {remote_audio} {remote_output_base}*"
            )
    
    def transcribe_batch(
        self,
        audio_files: List[Path],
        output_dir: Path,
        model: str = WHISPER_MODEL,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files in batch
        
        Args:
            audio_files: List of audio file paths
            output_dir: Output directory for transcripts
            model: Whisper model to use
            language: Language code
            
        Returns:
            List of transcription results
        """
        if not audio_files:
            return []
        
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        
        print(f"\\nTranscribing {len(audio_files)} files...")
        
        for i, audio_path in enumerate(audio_files, 1):
            print(f"\\n[{i}/{len(audio_files)}] Processing {audio_path.name}")
            
            try:
                result = self.transcribe_audio(
                    audio_path,
                    output_dir,
                    model,
                    language
                )
                results.append(result)
            except Exception as e:
                print(f"Error transcribing {audio_path.name}: {e}")
                results.append({
                    'error': str(e),
                    'audio_file': audio_path.name
                })
        
        # Summary
        successful = len([r for r in results if 'error' not in r])
        print(f"\\nBatch complete: {successful}/{len(audio_files)} files transcribed")
        
        return results
    
    def cleanup(self, destroy_instance: bool = False):
        """
        Clean up resources
        
        Args:
            destroy_instance: Whether to destroy the GPU instance
        """
        if destroy_instance and self.instance_manager.current_instance:
            self.instance_manager.destroy_current_instance()
        
        self.ssh_connection = None
    
    def _verify_setup(self):
        """Verify that the instance is properly set up"""
        # Check Python
        ret, out, err = self.ssh_connection.execute_command("python3 --version")
        if ret != 0:
            raise RuntimeError("Python not available on instance")
        
        # Check Whisper
        ret, out, err = self.ssh_connection.execute_command(
            "python3 -c 'import whisper; print(whisper.__version__)'"
        )
        if ret != 0:
            print("Installing Whisper...")
            self._install_dependencies()
        
        # Check GPU and CUDA
        print("Checking GPU...")
        ret, out, err = self.ssh_connection.execute_command("nvidia-smi")
        if ret != 0:
            print(f"nvidia-smi failed: {err}")
            raise RuntimeError("GPU not available on instance")
        else:
            print("nvidia-smi output:")
            print(out)
        
        # Test CUDA initialization
        print("Testing CUDA...")
        cuda_test = '''
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    # Initialize CUDA
    torch.cuda.init()
    # Create a small tensor to verify CUDA works
    x = torch.tensor([1.0]).cuda()
    print("CUDA initialization successful")
else:
    print("WARNING: CUDA not available!")
'''
        ret, out, err = self.ssh_connection.execute_command(
            f"python3 -c '{cuda_test}'", timeout=30
        )
        if ret != 0:
            print(f"CUDA test output: {out}")
            print(f"CUDA test error: {err}")
            print("WARNING: CUDA test failed, but continuing...")
        else:
            print(f"CUDA test output: {out}")
        
        print("Instance setup verified")
    
    def _install_dependencies(self):
        """Install required dependencies on the instance"""
        print("Installing dependencies...")
        
        install_script = """
        # Update system and install required packages
        apt-get update -qq
        apt-get install -y -qq ffmpeg
        
        # Upgrade pip
        pip install --upgrade pip
        
        # Install PyTorch with CUDA support first
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        
        # Install whisper packages
        pip install openai-whisper
        pip install faster-whisper
        """
        
        ret, out, err = self.ssh_connection.execute_command(
            install_script,
            stream_output=True
        )
        
        if ret != 0:
            raise RuntimeError(f"Failed to install dependencies: {err}")
    
    def _run_whisper(
        self,
        remote_audio: str,
        remote_output_base: str,
        model: str,
        language: Optional[str]
    ) -> str:
        """Run standard Whisper transcription"""
        cmd_parts = [
            "python3", "-m", "whisper",
            remote_audio,
            "--model", model,
            "--device", WHISPER_DEVICE,
            "--output_format", "json",
            "--output_dir", REMOTE_OUTPUT_DIR,
            "--verbose", "False"
        ]
        
        if language:
            cmd_parts.extend(["--language", language])
        
        cmd = " ".join(cmd_parts)
        
        print(f"Running Whisper {model} on GPU...")
        ret, out, err = self.ssh_connection.execute_command(
            cmd,
            stream_output=True,
            timeout=3600  # 1 hour timeout
        )
        
        if ret != 0:
            raise RuntimeError(f"Whisper failed: {err}")
        
        return f"{remote_output_base}.json"
    
    def _run_faster_whisper(
        self,
        remote_audio: str,
        remote_output_base: str,
        model: str,
        language: Optional[str]
    ) -> str:
        """Run faster-whisper transcription"""
        # Create Python script for faster-whisper
        script = f'''
import json
import time
import torch

# Initialize CUDA if available
if torch.cuda.is_available():
    print("Initializing CUDA...")
    torch.cuda.init()
    # Create a dummy tensor to ensure CUDA is working
    dummy = torch.tensor([1.0]).cuda()
    del dummy
    time.sleep(2)  # Give CUDA a moment
    print("CUDA initialized successfully")

from faster_whisper import WhisperModel

print("Loading WhisperModel...")
model = WhisperModel("{model}", device="{WHISPER_DEVICE}", compute_type="{WHISPER_COMPUTE_TYPE}")

segments, info = model.transcribe(
    "{remote_audio}",
    language={"'" + language + "'" if language else "None"},
    beam_size=5,
    word_timestamps=True
)

# Convert to standard format
result = {{
    "text": "",
    "segments": [],
    "language": info.language,
    "duration": info.duration
}}

for segment in segments:
    result["segments"].append({{
        "id": segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "words": [
            {{
                "start": word.start,
                "end": word.end,
                "word": word.word,
                "probability": word.probability
            }}
            for word in segment.words
        ] if segment.words else []
    }})
    result["text"] += segment.text + " "

result["text"] = result["text"].strip()

with open("{remote_output_base}.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Transcription complete. Duration: {{info.duration:.2f}}s")
'''
        
        # Create and run script
        script_path = f"{REMOTE_WORKSPACE}/transcribe_script.py"
        self.ssh_connection.create_remote_script(script, script_path)
        
        print(f"Running faster-whisper {model} on GPU...")
        ret, out, err = self.ssh_connection.execute_command(
            f"cd {REMOTE_WORKSPACE} && python3 transcribe_script.py",
            stream_output=True,
            timeout=3600
        )
        
        if ret != 0:
            # When streaming, errors are in stdout not stderr
            error_msg = out  # Since we're using stream_output=True
            raise RuntimeError(f"faster-whisper failed: {error_msg}")
        
        return f"{remote_output_base}.json"