"""
Simple Digital Ocean Runner - Run everything remotely with minimal complexity
"""
import time
from pathlib import Path
from typing import Dict, Optional, Any
import json

from .client import DigitalOceanClient
from .ssh_connection import DOSSHConnection
from .config import (
    DEFAULT_DROPLET_SIZE, DEFAULT_REGION, DEFAULT_IMAGE,
    SSH_KEY_NAME, LOCAL_SSH_KEY_PATH, SETUP_SCRIPT,
    REMOTE_WORKSPACE, REMOTE_CODE_DIR, DIGITAL_OCEAN_API_KEY
)


class SimpleDigitalOceanRunner:
    """
    Simple runner that:
    1. Spins up a single DO droplet
    2. Runs your code on it
    3. Returns results
    4. Cleans up
    
    No job queues, no thread pools, no complexity.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Use provided API key or fall back to config
        api_key = api_key or DIGITAL_OCEAN_API_KEY
        self.client = DigitalOceanClient(api_key)
        self.droplet = None
        self.ssh = None
        self.droplet_ip = None
    
    def start(self, droplet_size: str = DEFAULT_DROPLET_SIZE, wait_time: int = 60) -> str:
        """Start a single droplet and establish SSH connection"""
        print("Starting Digital Ocean droplet...")
        
        # Ensure SSH key exists
        ssh_key_id = self._ensure_ssh_key()
        
        # Create droplet
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.droplet = self.client.create_droplet(
            name=f"xtotext-{timestamp}",
            size=droplet_size,
            image=DEFAULT_IMAGE,
            region=DEFAULT_REGION,
            ssh_keys=[str(ssh_key_id)],
            tags=["xtotext", "temp"],
            user_data=SETUP_SCRIPT
        )
        
        print(f"Droplet {self.droplet['id']} created, waiting for it to be ready...")
        
        # Wait for active status
        if not self.client.wait_for_droplet_status(self.droplet['id'], 'active', timeout=300):
            raise Exception("Droplet failed to become active")
        
        # Get IP address
        self.droplet = self.client.get_droplet(self.droplet['id'])
        self.droplet_ip = self._get_droplet_ip()
        
        if not self.droplet_ip:
            raise Exception("No IP address found for droplet")
        
        print(f"Droplet ready at {self.droplet_ip}")
        
        # Wait a bit more for SSH to be ready
        print(f"Waiting {wait_time}s for SSH to be ready...")
        time.sleep(wait_time)
        
        # Establish SSH connection
        self.ssh = DOSSHConnection(self.droplet_ip)
        if not self.ssh.connect():
            raise Exception("Failed to establish SSH connection")
        
        print("SSH connection established")
        
        # Wait for setup script to complete
        print("Waiting for initial setup to complete...")
        wait_time = 0
        max_wait = 900  # 15 minutes max
        
        while wait_time < max_wait:
            # Check if setup is complete
            exit_code, _, _ = self.ssh.execute_command("test -f /workspace/.setup_complete")
            if exit_code == 0:
                print("Setup script completed successfully!")
                break
            
            # Check if cloud-init is still running
            exit_code, stdout, _ = self.ssh.execute_command("pgrep -f cloud-init")
            if exit_code == 0:
                print(f"Setup still running... ({wait_time}s elapsed)")
            else:
                # Cloud-init not running but no completion flag = setup failed
                raise Exception("Setup script failed to complete! Check /var/log/cloud-init-output.log on the droplet")
            
            time.sleep(10)
            wait_time += 10
        
        if wait_time >= max_wait:
            raise Exception(f"Setup script did not complete within {max_wait} seconds!")
        
        # Verify critical components
        exit_code, stdout, stderr = self.ssh.execute_command("which python3")
        if exit_code != 0:
            raise Exception("Python3 not installed - setup script failed!")
        
        return self.droplet_ip
    
    def upload_code(self, local_path: Path) -> None:
        """Upload your code to the droplet efficiently"""
        if not self.ssh:
            raise Exception("Not connected to droplet")
        
        print(f"Uploading {local_path.name} to droplet...")
        
        if local_path.is_dir():
            # Use tar for efficient directory upload
            import tempfile
            import subprocess
            
            # Create tar locally, excluding junk
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                tmp_path = Path(tmp.name)
                
                # Use system tar with exclusions
                exclude_args = []
                for pattern in ['__pycache__', '*.pyc', '.git', 'venv', '*.egg-info', '.pytest_cache']:
                    exclude_args.extend(['--exclude', pattern])
                
                cmd = ['tar', 'czf', str(tmp_path), '-C', str(local_path.parent)] + exclude_args + [local_path.name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"Failed to create tar: {result.stderr}")
                
                # Check size
                size_mb = tmp_path.stat().st_size / (1024 * 1024)
                print(f"  Created {size_mb:.1f}MB archive")
                
                # Upload tar
                remote_tar = f"/tmp/{local_path.name}.tar.gz"
                self.ssh.upload_file(tmp_path, remote_tar)
                
                # Extract on remote
                self.ssh.execute_command(f"mkdir -p {REMOTE_CODE_DIR}")
                self.ssh.execute_command(f"tar -xzf {remote_tar} -C {REMOTE_CODE_DIR}")
                self.ssh.execute_command(f"rm {remote_tar}")
                
                # Clean up
                tmp_path.unlink()
                
        else:
            # Single file upload
            remote_path = f"{REMOTE_CODE_DIR}/{local_path.name}"
            self.ssh.upload_file(local_path, remote_path)
        
        print(f"  Uploaded {local_path.name}")
    
    def run_command(self, command: str, timeout: int = 3600) -> Dict[str, Any]:
        """Run a command on the droplet"""
        if not self.ssh:
            raise Exception("Not connected to droplet")
        
        print(f"Running command: {command[:100]}...")
        
        start_time = time.time()
        exit_code, stdout, stderr = self.ssh.execute_command(command, timeout=timeout)
        duration = time.time() - start_time
        
        result = {
            'exit_code': exit_code,
            'stdout': stdout,
            'stderr': stderr,
            'duration': duration,
            'success': exit_code == 0
        }
        
        if exit_code == 0:
            print(f"Command completed successfully in {duration:.1f}s")
        else:
            print(f"Command failed with exit code {exit_code}")
            if stderr:
                print(f"Error: {stderr[:500]}")
        
        return result
    
    def run_transcription(self, audio_file: Path, output_dir: Path) -> Dict[str, Any]:
        """Run transcription on the droplet"""
        if not self.ssh:
            raise Exception("Not connected to droplet")
        
        # Upload audio file
        remote_audio = f"{REMOTE_WORKSPACE}/audio/{audio_file.name}"
        print(f"Uploading audio file {audio_file.name}...")
        self.ssh.create_directory(f"{REMOTE_WORKSPACE}/audio")
        self.ssh.upload_file(audio_file, remote_audio)
        
        # Run transcription
        command = f"""
cd {REMOTE_CODE_DIR}
source ../venv/bin/activate 2>/dev/null || python3 -m venv ../venv && source ../venv/bin/activate

# Install dependencies if needed
pip install openai-whisper faster-whisper yt-dlp

# Run transcription
python3 -c "
import whisper
import json
from pathlib import Path

print('Loading model...')
model = whisper.load_model('base')

print('Transcribing...')
result = model.transcribe('{remote_audio}')

# Save transcript
output_file = Path('{REMOTE_WORKSPACE}/results/transcript.json')
output_file.parent.mkdir(exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f'Transcript saved to {{output_file}}')
"
"""
        
        result = self.run_command(command)
        
        if result['success']:
            # Download transcript
            print("Downloading transcript...")
            output_dir.mkdir(exist_ok=True)
            local_transcript = output_dir / f"{audio_file.stem}_transcript.json"
            self.ssh.download_file(
                f"{REMOTE_WORKSPACE}/results/transcript.json",
                local_transcript
            )
            result['transcript_file'] = str(local_transcript)
        
        return result
    
    def run_batch_processing(self, audio_files: list[Path], output_dir: Path) -> list[Dict]:
        """Process multiple audio files"""
        results = []
        
        for i, audio_file in enumerate(audio_files):
            print(f"\nProcessing file {i+1}/{len(audio_files)}: {audio_file.name}")
            
            try:
                result = self.run_transcription(audio_file, output_dir)
                result['file'] = str(audio_file)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                results.append({
                    'file': str(audio_file),
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def download_results(self, remote_path: str, local_path: Path) -> None:
        """Download results from droplet"""
        if not self.ssh:
            raise Exception("Not connected to droplet")
        
        print(f"Downloading {remote_path} to {local_path}...")
        
        if self.ssh.directory_exists(remote_path):
            self.ssh.download_directory(remote_path, local_path)
        elif self.ssh.file_exists(remote_path):
            self.ssh.download_file(remote_path, local_path)
        else:
            raise Exception(f"Remote path {remote_path} not found")
    
    def cleanup(self, destroy_droplet: bool = True) -> None:
        """Clean up resources"""
        print("Cleaning up...")
        
        # Close SSH connection
        if self.ssh:
            self.ssh.close()
            self.ssh = None
        
        # Destroy droplet
        if destroy_droplet and self.droplet:
            print(f"Destroying droplet {self.droplet['id']}...")
            self.client.delete_droplet(self.droplet['id'])
            self.droplet = None
            self.droplet_ip = None
            print("Droplet destroyed")
    
    def _ensure_ssh_key(self) -> str:
        """Ensure SSH key exists in DO account"""
        existing_key = self.client.get_ssh_key_by_name(SSH_KEY_NAME)
        if existing_key:
            return existing_key["id"]
        
        # Read local SSH public key
        pub_key_path = LOCAL_SSH_KEY_PATH.with_suffix('.pub')
        if not pub_key_path.exists():
            raise FileNotFoundError(f"SSH public key not found at {pub_key_path}")
        
        public_key = pub_key_path.read_text().strip()
        
        # Create SSH key in DO
        new_key = self.client.create_ssh_key(SSH_KEY_NAME, public_key)
        return new_key["id"]
    
    def _get_droplet_ip(self) -> Optional[str]:
        """Get public IPv4 address of droplet"""
        for network in self.droplet.get("networks", {}).get("v4", []):
            if network["type"] == "public":
                return network["ip_address"]
        return None
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up on exit"""
        self.cleanup()


# Example usage function
def example_usage():
    """Example of how to use the simple runner"""
    
    # Initialize runner
    runner = SimpleDigitalOceanRunner()
    
    try:
        # Start droplet
        ip = runner.start(droplet_size="s-2vcpu-4gb")
        
        # Upload your code
        runner.upload_code(Path("/path/to/your/code"))
        
        # Run a simple command
        result = runner.run_command("cd /workspace && python3 your_script.py")
        
        # Or run transcription
        audio_file = Path("/path/to/audio.mp3")
        output_dir = Path("/path/to/output")
        result = runner.run_transcription(audio_file, output_dir)
        
        # Download any results
        runner.download_results("/workspace/results", Path("./local_results"))
        
    finally:
        # Always cleanup
        runner.cleanup()


# Even simpler: context manager usage
def example_with_context_manager():
    """Example using context manager for automatic cleanup"""
    
    with SimpleDigitalOceanRunner() as runner:
        runner.start()
        runner.upload_code(Path("./src"))
        result = runner.run_command("python3 /workspace/xtotext/main.py")
        print(result['stdout'])