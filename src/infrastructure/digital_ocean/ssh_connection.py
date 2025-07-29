"""
SSH connection handler for Digital Ocean droplets
"""
import paramiko
import time
import os
from pathlib import Path
from typing import Optional, Tuple, List

from .config import (
    SSH_USERNAME, SSH_PORT, SSH_TIMEOUT,
    CONNECTION_RETRIES, RETRY_DELAY,
    LOCAL_SSH_KEY_PATH
)


class DOSSHConnection:
    """Manages SSH connections to Digital Ocean droplets"""
    
    def __init__(self, host: str, port: int = SSH_PORT, username: str = SSH_USERNAME):
        self.host = host
        self.port = port
        self.username = username
        self.client = None
        self.sftp = None
        self.connected = False
    
    def connect(self, private_key_path: Optional[Path] = None) -> bool:
        """Establish SSH connection to droplet"""
        private_key_path = private_key_path or LOCAL_SSH_KEY_PATH
        
        if not private_key_path.exists():
            print(f"SSH private key not found at {private_key_path}")
            return False
        
        for attempt in range(CONNECTION_RETRIES):
            try:
                print(f"Connecting to {self.host}:{self.port} (attempt {attempt + 1}/{CONNECTION_RETRIES})")
                
                # Create SSH client
                self.client = paramiko.SSHClient()
                self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                # Load private key
                private_key = paramiko.RSAKey.from_private_key_file(str(private_key_path))
                
                # Connect
                self.client.connect(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    pkey=private_key,
                    timeout=SSH_TIMEOUT,
                    banner_timeout=SSH_TIMEOUT
                )
                
                # Create SFTP client
                self.sftp = self.client.open_sftp()
                
                self.connected = True
                print(f"Connected to {self.host}")
                return True
                
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < CONNECTION_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"Failed to connect to {self.host} after {CONNECTION_RETRIES} attempts")
                    return False
        
        return False
    
    def close(self) -> None:
        """Close SSH connection"""
        if self.sftp:
            try:
                self.sftp.close()
            except:
                pass
            self.sftp = None
        
        if self.client:
            try:
                self.client.close()
            except:
                pass
            self.client = None
        
        self.connected = False
    
    def is_connected(self) -> bool:
        """Check if connection is active"""
        if not self.connected or not self.client:
            return False
        
        try:
            # Test connection with a simple command
            self.client.exec_command('echo test', timeout=5)
            return True
        except:
            self.connected = False
            return False
    
    def execute_command(self, command: str, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        """Execute command on remote droplet"""
        if not self.is_connected():
            raise ConnectionError("Not connected to droplet")
        
        try:
            stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
            
            # Get exit status
            exit_status = stdout.channel.recv_exit_status()
            
            # Read output
            stdout_data = stdout.read().decode('utf-8')
            stderr_data = stderr.read().decode('utf-8')
            
            return exit_status, stdout_data, stderr_data
            
        except Exception as e:
            raise Exception(f"Command execution failed: {e}")
    
    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """Upload file to droplet"""
        if not self.is_connected():
            raise ConnectionError("Not connected to droplet")
        
        try:
            # Create remote directory if needed
            remote_dir = os.path.dirname(remote_path)
            if remote_dir:
                self.execute_command(f"mkdir -p {remote_dir}")
            
            # Upload file
            self.sftp.put(str(local_path), remote_path)
            print(f"Uploaded {local_path} to {remote_path}")
            
        except Exception as e:
            raise Exception(f"File upload failed: {e}")
    
    def download_file(self, remote_path: str, local_path: Path) -> None:
        """Download file from droplet"""
        if not self.is_connected():
            raise ConnectionError("Not connected to droplet")
        
        try:
            # Create local directory if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            self.sftp.get(remote_path, str(local_path))
            print(f"Downloaded {remote_path} to {local_path}")
            
        except Exception as e:
            raise Exception(f"File download failed: {e}")
    
    def upload_directory(self, local_dir: Path, remote_dir: str) -> None:
        """Upload entire directory to droplet"""
        if not self.is_connected():
            raise ConnectionError("Not connected to droplet")
        
        # Create remote directory
        self.execute_command(f"mkdir -p {remote_dir}")
        
        # Define files/dirs to skip
        skip_patterns = {
            '__pycache__', '.pyc', '.pyo', '.pytest_cache', 
            '.git', '.gitignore', '.DS_Store', '*.egg-info',
            'venv', 'env', '.env', 'node_modules'
        }
        
        # Walk through local directory
        file_count = 0
        skipped = 0
        
        for item in local_dir.rglob("*"):
            # Skip if any part of the path contains skip patterns
            if any(pattern in str(item) for pattern in skip_patterns):
                skipped += 1
                continue
                
            # Skip if file extension matches skip patterns
            if any(str(item).endswith(pattern) for pattern in skip_patterns if pattern.startswith('.')):
                skipped += 1
                continue
            
            if item.is_file():
                relative_path = item.relative_to(local_dir)
                remote_path = f"{remote_dir}/{relative_path}"
                self.upload_file(item, remote_path)
                file_count += 1
                
                if file_count % 10 == 0:
                    print(f"  Uploaded {file_count} files...")
        
        print(f"  Upload complete: {file_count} files uploaded, {skipped} skipped")
    
    def download_directory(self, remote_dir: str, local_dir: Path) -> None:
        """Download entire directory from droplet"""
        if not self.is_connected():
            raise ConnectionError("Not connected to droplet")
        
        # Create local directory
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # List remote directory contents
        exit_status, stdout, _ = self.execute_command(f"find {remote_dir} -type f")
        
        if exit_status == 0:
            files = stdout.strip().split('\n')
            for remote_file in files:
                if remote_file:
                    relative_path = remote_file.replace(f"{remote_dir}/", "")
                    local_path = local_dir / relative_path
                    self.download_file(remote_file, local_path)
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists on droplet"""
        if not self.is_connected():
            raise ConnectionError("Not connected to droplet")
        
        exit_status, _, _ = self.execute_command(f"test -f {remote_path}")
        return exit_status == 0
    
    def directory_exists(self, remote_path: str) -> bool:
        """Check if directory exists on droplet"""
        if not self.is_connected():
            raise ConnectionError("Not connected to droplet")
        
        exit_status, _, _ = self.execute_command(f"test -d {remote_path}")
        return exit_status == 0
    
    def create_directory(self, remote_path: str) -> None:
        """Create directory on droplet"""
        if not self.is_connected():
            raise ConnectionError("Not connected to droplet")
        
        self.execute_command(f"mkdir -p {remote_path}")
    
    def delete_file(self, remote_path: str) -> None:
        """Delete file on droplet"""
        if not self.is_connected():
            raise ConnectionError("Not connected to droplet")
        
        self.execute_command(f"rm -f {remote_path}")
    
    def delete_directory(self, remote_path: str) -> None:
        """Delete directory on droplet"""
        if not self.is_connected():
            raise ConnectionError("Not connected to droplet")
        
        self.execute_command(f"rm -rf {remote_path}")
    
    def list_directory(self, remote_path: str) -> List[str]:
        """List directory contents on droplet"""
        if not self.is_connected():
            raise ConnectionError("Not connected to droplet")
        
        exit_status, stdout, _ = self.execute_command(f"ls -la {remote_path}")
        
        if exit_status == 0:
            return stdout.strip().split('\n')
        return []
    
    def get_file_size(self, remote_path: str) -> int:
        """Get size of file on droplet"""
        if not self.is_connected():
            raise ConnectionError("Not connected to droplet")
        
        exit_status, stdout, _ = self.execute_command(f"stat -c%s {remote_path}")
        
        if exit_status == 0:
            return int(stdout.strip())
        return 0
    
    def tail_file(self, remote_path: str, lines: int = 100) -> str:
        """Tail a file on droplet"""
        if not self.is_connected():
            raise ConnectionError("Not connected to droplet")
        
        exit_status, stdout, _ = self.execute_command(f"tail -n {lines} {remote_path}")
        
        if exit_status == 0:
            return stdout
        return ""
    
    def execute_script(self, script_content: str, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        """Execute multi-line script on droplet"""
        if not self.is_connected():
            raise ConnectionError("Not connected to droplet")
        
        # Create temporary script file
        script_path = f"/tmp/script_{int(time.time())}.sh"
        
        # Upload script
        self.sftp.file(script_path, 'w').write(script_content)
        
        # Make executable
        self.execute_command(f"chmod +x {script_path}")
        
        # Execute script
        exit_status, stdout, stderr = self.execute_command(f"bash {script_path}", timeout=timeout)
        
        # Clean up
        self.delete_file(script_path)
        
        return exit_status, stdout, stderr