"""
Vast.ai Instance Manager for Transcription
"""
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
from pathlib import Path
import time

from .client import VastAIClient
from .config import (
    DEFAULT_GPU_TYPE, DEFAULT_GPU_COUNT, DEFAULT_MIN_GPU_RAM,
    DEFAULT_MIN_RAM, DEFAULT_MIN_DISK, DEFAULT_MAX_PRICE,
    SETUP_SCRIPT
)


class InstanceManager:
    """Manages Vast.ai instances for transcription tasks"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = VastAIClient(api_key)
        self.current_instance = None
        
        # Instance state file
        self.state_file = Path.home() / ".xtotext" / "vast_instance.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing instance if any
        self._load_state()
    
    def find_best_instance(
        self,
        gpu_type: str = DEFAULT_GPU_TYPE,
        min_gpu_ram: int = DEFAULT_MIN_GPU_RAM,
        min_ram: int = DEFAULT_MIN_RAM,
        min_disk: int = DEFAULT_MIN_DISK,
        max_price: float = DEFAULT_MAX_PRICE,
        exclude_countries: List[str] = None,
        prefer_verified: bool = True,
        min_reliability: float = 98.0
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best available GPU instance for transcription
        
        Args:
            gpu_type: GPU model to search for
            min_gpu_ram: Minimum GPU RAM in GB
            min_ram: Minimum system RAM in GB
            min_disk: Minimum disk space in GB
            max_price: Maximum price per hour
            exclude_countries: List of country codes to exclude (e.g., ['CN'])
            prefer_verified: Whether to prefer verified instances
            min_reliability: Minimum reliability percentage
            
        Returns:
            Best instance offer or None if no suitable instances found
        """
        if exclude_countries is None:
            # Exclude countries where connections are problematic
            exclude_countries = ['CN', 'BG', 'RO']  # China, Bulgaria, Romania
            
        print(f"Searching for {gpu_type} instances...")
        
        # First try verified instances
        if prefer_verified:
            offers = self.client.search_instances(
                gpu_type=gpu_type,
                min_gpu_ram=min_gpu_ram,
                min_ram=min_ram,
                min_disk=min_disk,
                max_price=max_price,
                verified_only=True
            )
        else:
            offers = self.client.search_instances(
                gpu_type=gpu_type,
                min_gpu_ram=min_gpu_ram,
                min_ram=min_ram,
                min_disk=min_disk,
                max_price=max_price
            )
        
        # Filter out excluded countries and low reliability
        filtered_offers = []
        for offer in offers:
            location = offer.get('geolocation', '')
            reliability = offer.get('reliability2', offer.get('reliability', 0)) * 100
            
            # Check country exclusion
            exclude = False
            for country in exclude_countries:
                if country in location:
                    exclude = True
                    break
            
            if not exclude and reliability >= min_reliability:
                filtered_offers.append(offer)
        
        if not filtered_offers:
            print(f"No suitable {gpu_type} instances found")
            if prefer_verified and not offers:
                print("Trying unverified instances...")
                return self.find_best_instance(
                    gpu_type=gpu_type,
                    min_gpu_ram=min_gpu_ram,
                    min_ram=min_ram,
                    min_disk=min_disk,
                    max_price=max_price,
                    exclude_countries=exclude_countries,
                    prefer_verified=False,
                    min_reliability=min_reliability
                )
            return None
        
        # Sort by price and reliability
        filtered_offers.sort(key=lambda x: (
            x.get("dph_total", float('inf')),
            -x.get("reliability2", x.get("reliability", 0))
        ))
        
        best_offer = filtered_offers[0]
        print(f"Found best {gpu_type} instance:")
        print(f"  - Price: ${best_offer.get('dph_total', 0):.3f}/hour")
        print(f"  - Location: {best_offer.get('geolocation', 'Unknown')}")
        print(f"  - GPU RAM: {best_offer.get('gpu_ram', 0) / 1024:.1f} GB")
        print(f"  - GPU Fraction: {best_offer.get('gpu_frac', 1.0) * 100:.0f}%")
        print(f"  - Num GPUs: {best_offer.get('num_gpus', 1)}")
        print(f"  - System RAM: {best_offer.get('cpu_ram', 0) / 1024:.1f} GB")
        print(f"  - Disk: {best_offer.get('disk_space', 0)} GB")
        print(f"  - Reliability: {best_offer.get('reliability2', best_offer.get('reliability', 0)) * 100:.1f}%")
        print(f"  - Status: {'Verified' if best_offer.get('verification') == 'verified' else 'Unverified'}")
        
        return best_offer
    
    def create_transcription_instance(
        self,
        gpu_type: str = DEFAULT_GPU_TYPE,
        max_price: float = DEFAULT_MAX_PRICE,
        disk_size: int = DEFAULT_MIN_DISK,
        exclude_countries: List[str] = None,
        ubuntu_version: str = "22.04",
        ssh_key_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Create a new GPU instance optimized for transcription
        
        Args:
            gpu_type: GPU model to use
            max_price: Maximum price per hour
            disk_size: Disk size in GB
            exclude_countries: Countries to exclude (default: ['CN'])
            ubuntu_version: Ubuntu version to use (20.04 or 22.04)
            ssh_key_path: Path to SSH public key (default: ~/.ssh/id_rsa.pub)
            
        Returns:
            Instance details
        """
        # Check if we already have a valid instance
        existing = self.get_current_instance()
        if existing:
            print(f"Already have instance {existing['id']} running")
            return existing
        
        # Find best offer
        offer = self.find_best_instance(
            gpu_type=gpu_type,
            max_price=max_price,
            exclude_countries=exclude_countries
        )
        
        if not offer:
            raise RuntimeError(f"No suitable {gpu_type} instances available")
        
        print(f"\\nCreating instance from offer {offer['id']}...")
        
        # Use the template that has the right cuDNN version for faster-whisper
        template_hash = "b09bff7878753196c7e69a0bf2b916b8"  # Template ID 261658
        
        # Create instance with SSH enabled
        response = self.client.create_instance(
            offer_id=offer['id'],
            template_hash=template_hash,
            disk_size=disk_size,
            ssh=True,
            direct=True  # Use direct connection for better performance
        )
        
        instance_id = response.get("new_instance_id") or response.get("instance_id") or response.get("new_contract")
        if not instance_id:
            raise RuntimeError(f"Failed to get instance ID from response: {response}")
        
        print(f"Instance {instance_id} created")
        
        print(f"Waiting for instance {instance_id} to start...")
        print("Note: GPU instances typically take 3-5 minutes to start")
        print("Will kill and retry if it takes more than 5 minutes...")
        
        # Wait for instance to be ready (5 minutes timeout)
        try:
            instance = self.client.wait_for_instance(instance_id, timeout=300)  # 5 minutes
        except RuntimeError as e:
            if "timeout" in str(e).lower():
                print("\nInstance took too long to start (>5 minutes), destroying and retrying...")
                
                # Destroy the slow instance
                try:
                    self.client.destroy_instance(instance_id)
                    print("Slow instance destroyed")
                except Exception as destroy_error:
                    print(f"Warning: Failed to destroy slow instance: {destroy_error}")
                
                # Wait a bit before retrying
                time.sleep(10)
                
                # Try once more with a different instance
                print("\nRetrying with a new instance...")
                new_instance_id = self._select_and_start_instance(
                    offers, 
                    gpu_type=gpu_type, 
                    max_price=max_price,
                    num_gpus=num_gpus,
                    min_gpu_ram=min_gpu_ram,
                    min_ram=min_ram,
                    min_disk=min_disk
                )
                
                # Wait again, but this time let it fail if it times out
                instance = self.client.wait_for_instance(new_instance_id, timeout=300)
            else:
                raise
        
        # Attach SSH key AFTER instance is running
        if ssh_key_path is None:
            ssh_key_path = Path.home() / ".ssh" / "id_rsa.pub"
        
        if ssh_key_path.exists():
            print(f"Attaching SSH key from {ssh_key_path}...")
            try:
                ssh_key = ssh_key_path.read_text().strip()
                print(f"  SSH key length: {len(ssh_key)} chars")
                print(f"  SSH key preview: {ssh_key[:50]}...")
                
                result = self.client.attach_ssh_key(instance_id, ssh_key)
                print(f"  API response: {result}")
                
                # Verify it actually worked
                if not result.get('success'):
                    raise RuntimeError(f"SSH key attachment failed: {result}")
                    
                print("SSH key attached successfully")
                
                # Double check by getting instance details again
                time.sleep(2)
                instance = self.client.get_instance(instance_id)
                print(f"  Instance SSH keys: {instance.get('ssh_keys', 'None')}")
            except Exception as e:
                print(f"ERROR: Failed to attach SSH key: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Cannot continue without SSH key: {e}")
        
        # Extract direct connection details if available
        public_ip = instance.get('public_ipaddr')
        ports = instance.get('ports', {})
        
        # Look for direct SSH port
        direct_ssh_port = None
        if isinstance(ports, dict) and '22/tcp' in ports:
            port_mappings = ports.get('22/tcp', [])
            if isinstance(port_mappings, list) and len(port_mappings) > 0:
                first_mapping = port_mappings[0]
                if isinstance(first_mapping, dict) and 'HostPort' in first_mapping:
                    direct_ssh_port = int(first_mapping['HostPort'])
        
        # Prefer direct connection if available
        if public_ip and direct_ssh_port:
            print(f"Found direct connection: {public_ip}:{direct_ssh_port}")
            instance['direct_ssh_host'] = public_ip
            instance['direct_ssh_port'] = direct_ssh_port
            
            # Keep proxy as fallback
            instance['proxy_ssh_host'] = instance.get('ssh_host')
            instance['proxy_ssh_port'] = instance.get('ssh_port')
            
            # Use direct by default
            instance['ssh_host'] = public_ip
            instance['ssh_port'] = direct_ssh_port
            instance['is_direct'] = True
        else:
            # Only proxy available
            instance['is_direct'] = False
        
        print(f"Instance {instance_id} is ready!")
        print(f"  - SSH: {instance['ssh_host']}:{instance['ssh_port']}")
        print(f"  - Cost: ${instance.get('dph_total', instance.get('dph', 0)):.3f}/hour")
        print(f"  - Connection type: {'Direct' if instance.get('is_direct') else 'Proxy'}")
        
        # Debug: show more connection details
        if 'vast.ai' in instance.get('ssh_host', ''):
            print(f"  - NOTE: This is a proxy connection through Vast.ai servers")
        
        # Save instance state
        self.current_instance = instance
        self._save_state()
        
        return instance
    
    def get_current_instance(self) -> Optional[Dict[str, Any]]:
        """Get current instance details if one exists"""
        if not self.current_instance:
            return None
        
        try:
            # Refresh instance details from API
            instance = self.client.get_instance(self.current_instance['id'])
            
            # Check if instance is actually running
            actual_status = instance.get('actual_status') or instance.get('status', {}).get('state')
            if actual_status not in ['running', 'loading']:
                print(f"Instance {self.current_instance['id']} is not running (status: {actual_status})")
                self.current_instance = None
                self._save_state()
                return None
                
            self.current_instance = instance
            self._save_state()
            return instance
        except Exception as e:
            # Instance might be gone or API error
            print(f"Failed to get instance {self.current_instance['id']}: {e}")
            self.current_instance = None
            self._save_state()
            return None
    
    def destroy_current_instance(self) -> bool:
        """Destroy the current instance"""
        if not self.current_instance:
            print("No instance to destroy")
            return False
        
        instance_id = self.current_instance['id']
        print(f"Destroying instance {instance_id}...")
        
        success = self.client.destroy_instance(instance_id)
        
        if success:
            print(f"Instance {instance_id} destroyed")
            self.current_instance = None
            self._save_state()
        else:
            print(f"Failed to destroy instance {instance_id}")
        
        return success
    
    def estimate_cost(self, duration_minutes: int) -> float:
        """
        Estimate transcription cost based on instance price
        
        Args:
            duration_minutes: Estimated transcription duration
            
        Returns:
            Estimated cost in USD
        """
        if not self.current_instance:
            # Use default price for estimation
            price_per_hour = DEFAULT_MAX_PRICE
        else:
            price_per_hour = self.current_instance.get('dph_total', self.current_instance.get('dph', DEFAULT_MAX_PRICE))
        
        hours = duration_minutes / 60
        return price_per_hour * hours
    
    def get_instance_uptime(self) -> Optional[int]:
        """Get current instance uptime in minutes"""
        if not self.current_instance:
            return None
        
        start_time = self.current_instance.get('start_time')
        if not start_time:
            return None
        
        # Parse start time
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        uptime = datetime.now() - start_dt
        
        return int(uptime.total_seconds() / 60)
    
    def get_instance_cost(self) -> Optional[float]:
        """Get current instance cost based on uptime"""
        uptime_minutes = self.get_instance_uptime()
        if not uptime_minutes:
            return None
        
        return self.estimate_cost(uptime_minutes)
    
    def _save_state(self):
        """Save current instance state to file"""
        state = {
            "instance": self.current_instance,
            "updated_at": datetime.now().isoformat()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load instance state from file"""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.current_instance = state.get("instance")
            
            # Verify instance still exists
            if self.current_instance:
                try:
                    self.client.get_instance(self.current_instance['id'])
                except Exception:
                    # Instance no longer exists
                    self.current_instance = None
                    self._save_state()
        except Exception:
            # Invalid state file
            self.current_instance = None