"""
SSH Connection Management for Vast.ai Instances
"""
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import tempfile
import os

from .config import (
    SSH_KEY_PATH, SSH_USERNAME, SSH_TIMEOUT,
    CONNECTION_RETRIES, RETRY_DELAY
)


class SSHConnection:
    """Manages SSH connections to Vast.ai instances"""
    
    def __init__(
        self,
        host: str,
        port: int,
        username: str = SSH_USERNAME,
        key_path: Optional[Path] = None
    ):
        self.host = host
        self.port = port
        self.username = username
        self.key_path = key_path or SSH_KEY_PATH
        
        if not self.key_path.exists():
            raise FileNotFoundError(f"SSH key not found: {self.key_path}")
        
        # Base SSH command
        self.ssh_base = [
            "ssh",
            "-i", str(self.key_path),
            "-p", str(self.port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={SSH_TIMEOUT}",
            "-o", "ServerAliveInterval=60",
            "-o", "ServerAliveCountMax=3",
            f"{self.username}@{self.host}"
        ]
        
        # SCP base command
        self.scp_base = [
            "scp",
            "-i", str(self.key_path),
            "-P", str(self.port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={SSH_TIMEOUT}"
        ]
    
    def test_connection(self) -> bool:
        """Test if SSH connection works"""
        try:
            # Add verbose flag for debugging
            test_cmd = self.ssh_base.copy()
            # Insert -v flag before the command
            test_cmd.insert(1, "-v")
            test_cmd.append("echo 'Connection test'")
            
            print(f"  Debug: Testing SSH to {self.host}:{self.port}")
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"  SSH command failed with code {result.returncode}")
                if result.stderr:
                    # Show key debug info from stderr
                    for line in result.stderr.split('\n'):
                        if any(keyword in line.lower() for keyword in ['connect', 'authent', 'key', 'denied', 'closed']):
                            print(f"  Debug: {line.strip()}")
            
            return result.returncode == 0 and "Connection test" in result.stdout
        except Exception as e:
            print(f"  SSH test failed: {e}")
            return False
    
    def wait_for_connection(
        self,
        timeout: int = 300,
        check_interval: int = 10
    ) -> bool:
        """
        Wait for SSH connection to become available
        
        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Time between connection attempts
            
        Returns:
            True if connection established, False if timeout
        """
        print(f"Waiting for SSH connection to {self.host}:{self.port}...")
        
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < timeout:
            attempt += 1
            elapsed = int(time.time() - start_time)
            print(f"  Attempt {attempt} (elapsed: {elapsed}s)...")
            
            if self.test_connection():
                print(f"✓ SSH connection established after {attempt} attempts")
                return True
            
            # Increase wait time between attempts (10s, 20s, 30s, then 30s each)
            wait_time = min(attempt * 10, 30)
            print(f"  Waiting {wait_time}s before next attempt...")
            time.sleep(wait_time)
        
        return False
    
    def execute_command(
        self,
        command: str,
        timeout: Optional[int] = None,
        stream_output: bool = False
    ) -> Tuple[int, str, str]:
        """
        Execute command on remote instance
        
        Args:
            command: Command to execute
            timeout: Command timeout in seconds
            stream_output: Stream output in real-time
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        cmd = self.ssh_base + [command]
        
        if stream_output:
            # Stream output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            output_lines = []
            for line in process.stdout:
                print(line, end='')
                output_lines.append(line)
            
            process.wait()
            return process.returncode, ''.join(output_lines), ''
        else:
            # Capture output
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                return result.returncode, result.stdout, result.stderr
            except subprocess.TimeoutExpired:
                raise TimeoutError(f"Command timed out after {timeout} seconds")
    
    def upload_file(
        self,
        local_path: Path,
        remote_path: str,
        show_progress: bool = True
    ) -> bool:
        """
        Upload file to remote instance
        
        Args:
            local_path: Local file path
            remote_path: Remote destination path
            show_progress: Show transfer progress
            
        Returns:
            True if successful, False otherwise
        """
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        # Add progress flag if requested
        cmd = self.scp_base.copy()
        if show_progress:
            cmd.insert(1, "-v")
        
        cmd.extend([
            str(local_path),
            f"{self.username}@{self.host}:{remote_path}"
        ])
        
        print(f"Uploading {local_path.name} to {remote_path}...")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Upload complete")
            return True
        else:
            print(f"Upload failed: {result.stderr}")
            return False
    
    def download_file(
        self,
        remote_path: str,
        local_path: Path,
        show_progress: bool = True
    ) -> bool:
        """
        Download file from remote instance
        
        Args:
            remote_path: Remote file path
            local_path: Local destination path
            show_progress: Show transfer progress
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure local directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add progress flag if requested
        cmd = self.scp_base.copy()
        if show_progress:
            cmd.insert(1, "-v")
        
        cmd.extend([
            f"{self.username}@{self.host}:{remote_path}",
            str(local_path)
        ])
        
        print(f"Downloading {remote_path} to {local_path}...")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Download complete")
            return True
        else:
            print(f"Download failed: {result.stderr}")
            return False
    
    def upload_directory(
        self,
        local_dir: Path,
        remote_dir: str,
        show_progress: bool = True
    ) -> bool:
        """
        Upload directory to remote instance
        
        Args:
            local_dir: Local directory path
            remote_dir: Remote destination directory
            show_progress: Show transfer progress
            
        Returns:
            True if successful, False otherwise
        """
        if not local_dir.exists() or not local_dir.is_dir():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")
        
        # Create remote directory
        self.execute_command(f"mkdir -p {remote_dir}")
        
        # Use rsync for efficient directory transfer
        cmd = [
            "rsync",
            "-avz",
            "--progress" if show_progress else "",
            "-e", f"ssh -i {self.key_path} -p {self.port} -o StrictHostKeyChecking=no",
            str(local_dir) + "/",
            f"{self.username}@{self.host}:{remote_dir}/"
        ]
        
        # Remove empty string if no progress
        cmd = [c for c in cmd if c]
        
        print(f"Uploading directory {local_dir.name} to {remote_dir}...")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Directory upload complete")
            return True
        else:
            print(f"Directory upload failed: {result.stderr}")
            return False
    
    def create_remote_script(
        self,
        script_content: str,
        remote_path: str,
        make_executable: bool = True
    ) -> bool:
        """
        Create a script file on the remote instance
        
        Args:
            script_content: Script content
            remote_path: Remote script path
            make_executable: Make script executable
            
        Returns:
            True if successful, False otherwise
        """
        # Create temporary local file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(script_content)
            temp_path = Path(f.name)
        
        try:
            # Upload script
            success = self.upload_file(temp_path, remote_path, show_progress=False)
            
            if success and make_executable:
                # Make executable
                self.execute_command(f"chmod +x {remote_path}")
            
            return success
        finally:
            # Clean up temp file
            temp_path.unlink()
    
    def run_with_retries(
        self,
        command: str,
        retries: int = CONNECTION_RETRIES,
        delay: int = RETRY_DELAY
    ) -> Tuple[int, str, str]:
        """
        Execute command with automatic retries on failure
        
        Args:
            command: Command to execute
            retries: Number of retry attempts
            delay: Delay between retries in seconds
            
        Returns:
            Command output tuple
        """
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                return self.execute_command(command)
            except Exception as e:
                last_error = e
                if attempt < retries:
                    print(f"Command failed (attempt {attempt + 1}/{retries + 1}), retrying in {delay}s...")
                    time.sleep(delay)
        
        raise last_error