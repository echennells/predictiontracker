"""
Vast.ai API Client
"""
import requests
from typing import Dict, List, Optional, Any
import json
import time

from .config import VAST_API_KEY, VAST_API_BASE


class VastAIClient:
    """Client for interacting with Vast.ai API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or VAST_API_KEY
        if not self.api_key:
            raise ValueError("Vast.ai API key not provided. Set VAST_API_KEY environment variable.")
        
        self.session = requests.Session()
        # Try different auth header format
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json"
        })
        # Add API key as a parameter instead
        self.api_key_param = {"api_key": self.api_key}
    
    def search_instances(
        self,
        gpu_type: str = "RTX 3080",
        min_gpu_ram: int = 10,
        min_ram: int = 16,
        min_disk: int = 50,
        max_price: float = 1.00,
        limit: int = 20,
        verified_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for available GPU instances
        
        Args:
            gpu_type: GPU model to search for
            min_gpu_ram: Minimum GPU RAM in GB
            min_ram: Minimum system RAM in GB 
            min_disk: Minimum disk space in GB
            max_price: Maximum price per hour in USD
            limit: Maximum number of results
            
        Returns:
            List of available instances matching criteria
        """
        # Build search query - find cheapest globally
        query = {
            "gpu_name": {"eq": gpu_type},
            "gpu_ram": {"gte": min_gpu_ram * 1024},  # Convert to MB
            "cpu_ram": {"gte": min_ram * 1024},  # Convert to MB
            "disk_space": {"gte": min_disk},
            "dph_total": {"lte": max_price},
            "rentable": {"eq": True},
            "num_gpus": {"gte": 1},  # Full GPU, not fractional
            "gpu_frac": {"gte": 0.99},  # At least 99% of GPU
            "order": [["dph_total", "asc"]],  # Sort by total price ascending
            "type": "on-demand"
        }
        
        if verified_only:
            query["verification"] = {"eq": "verified"}
        
        params = {
            "q": json.dumps(query),
            "limit": limit,
            **self.api_key_param  # Add API key to params
        }
        
        response = self.session.get(
            f"{VAST_API_BASE}/bundles",
            params=params
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")
        
        data = response.json()
        return data.get("offers", [])
    
    def create_instance(
        self,
        offer_id: int,
        image: Optional[str] = None,
        template_hash: Optional[str] = None,
        disk_size: int = 50,
        onstart_script: Optional[str] = None,
        ssh: bool = True,
        direct: bool = True,
        jupyter: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new GPU instance
        
        Args:
            offer_id: ID of the offer to rent
            image: Docker image to use (if not using template)
            template_hash: Template hash to use (preferred over image)
            disk_size: Disk size in GB
            onstart_script: Script to run on instance start
            ssh: Enable SSH access
            direct: Use direct connection for SSH
            jupyter: Enable Jupyter notebook
            
        Returns:
            Instance creation response
        """
        payload = {
            "id": offer_id,
            "disk": disk_size
        }
        
        # Use template hash if provided, otherwise use image
        if template_hash:
            payload["template_hash_id"] = template_hash
            # Don't include image field when using template_hash
        elif image:
            payload["image"] = image
        else:
            raise ValueError("Either template_hash or image must be provided")
        
        # Set instance type
        if ssh:
            payload["ssh"] = True
        if jupyter:
            payload["jupyter"] = True
        if direct:
            payload["direct"] = True
            
        if onstart_script:
            payload["onstart"] = onstart_script
        
        # Use the API key parameter
        params = {**self.api_key_param}
        
        response = self.session.put(
            f"{VAST_API_BASE}/asks/{offer_id}/",
            json=payload,
            params=params
        )
        
        if response.status_code not in [200, 201]:
            raise RuntimeError(f"Failed to create instance: {response.text}")
        
        return response.json()
    
    def get_instance(self, instance_id: int) -> Dict[str, Any]:
        """Get instance details"""
        params = {**self.api_key_param}
        response = self.session.get(
            f"{VAST_API_BASE}/instances/{instance_id}/",
            params=params
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get instance: {response.text}")
        
        data = response.json()
        # The API returns the instance wrapped in an "instances" key
        if "instances" in data and isinstance(data["instances"], dict):
            return data["instances"]
        return data
    
    def get_instances(self) -> List[Dict[str, Any]]:
        """Get all instances for the account"""
        params = {**self.api_key_param}
        response = self.session.get(f"{VAST_API_BASE}/instances/", params=params)
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get instances: {response.text}")
        
        data = response.json()
        return data.get("instances", [])
    
    def destroy_instance(self, instance_id: int) -> bool:
        """Destroy an instance"""
        params = {**self.api_key_param}
        response = self.session.delete(
            f"{VAST_API_BASE}/instances/{instance_id}/",
            params=params
        )
        
        return response.status_code == 200
    
    def attach_ssh_key(self, instance_id: int, ssh_key: str) -> Dict[str, Any]:
        """
        Attach SSH key to instance
        
        Args:
            instance_id: Instance ID
            ssh_key: SSH public key content
            
        Returns:
            API response
        """
        payload = {
            "ssh_key": ssh_key
        }
        
        params = {**self.api_key_param}
        
        response = self.session.post(
            f"{VAST_API_BASE}/instances/{instance_id}/ssh/",
            json=payload,
            params=params
        )
        
        if response.status_code not in [200, 201]:
            raise RuntimeError(f"Failed to attach SSH key: {response.text}")
        
        # Handle empty or non-JSON responses
        if not response.text:
            return {"success": True, "message": "SSH key attached"}
        
        try:
            return response.json()
        except json.JSONDecodeError:
            # If response is not JSON, return a success indicator if status is good
            if response.status_code in [200, 201]:
                return {"success": True, "message": "SSH key attached", "raw_response": response.text}
            else:
                raise RuntimeError(f"Invalid response from attach_ssh_key: {response.text}")
    
    def wait_for_instance(
        self, 
        instance_id: int, 
        timeout: int = 420,  # 7 minutes total
        check_interval: int = 10  # kept for backwards compatibility but not used
    ) -> Dict[str, Any]:
        """
        Wait for instance to be ready with intelligent backoff
        
        Args:
            instance_id: Instance ID to wait for
            timeout: Maximum time to wait in seconds (default 7 minutes)
            check_interval: Deprecated, kept for backwards compatibility
            
        Returns:
            Instance details when ready
        """
        start_time = time.time()
        
        # Intelligent check intervals: 10s, 20s, 30s, 60s, 90s, 120s, then every 60s
        check_intervals = [10, 20, 30, 60, 90, 120]
        interval_index = 0
        last_check = 0
        
        print(f"Waiting for instance {instance_id} to start...")
        
        while time.time() - start_time < timeout:
            current_time = time.time() - start_time
            
            # Determine next check interval
            if interval_index < len(check_intervals):
                next_interval = check_intervals[interval_index]
            else:
                next_interval = 60  # Check every minute after initial intervals
            
            # Only check if enough time has passed
            if current_time - last_check >= next_interval or last_check == 0:
                elapsed_mins = current_time / 60
                print(f"  Checking instance status... ({elapsed_mins:.1f} minutes elapsed)")
                
                try:
                    instance = self.get_instance(instance_id)
                    status = instance.get("status", {})
                    actual_status = instance.get("actual_status") or status.get("state")
                    
                    print(f"  Status: {actual_status}")
                    
                    if actual_status == "running":
                        # Check if SSH is available
                        ssh_host = instance.get("ssh_host")
                        ssh_port = instance.get("ssh_port") 
                        
                        if ssh_host and ssh_port:
                            print(f"✓ Instance ready! SSH: {ssh_host}:{ssh_port}")
                            return instance
                        else:
                            print("  Instance running but SSH not yet available...")
                            
                except Exception as e:
                    print(f"  Error checking status: {e}")
                
                last_check = current_time
                interval_index += 1
            
            # Short sleep to prevent tight loop
            time.sleep(1)
        
        raise TimeoutError(f"Instance {instance_id} not ready after {timeout} seconds")
    
    def _get_ssh_pubkey(self) -> str:
        """Get SSH public key for instance access"""
        from .config import SSH_KEY_PATH
        
        pubkey_path = SSH_KEY_PATH.with_suffix(".pub")
        if not pubkey_path.exists():
            raise FileNotFoundError(f"SSH public key not found: {pubkey_path}")
        
        return pubkey_path.read_text().strip()