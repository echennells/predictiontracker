"""
Digital Ocean API client for managing droplets and resources
"""
import requests
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from .config import DIGITAL_OCEAN_API_KEY, DIGITAL_OCEAN_API_BASE


class DigitalOceanClient:
    """Client for interacting with Digital Ocean API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DIGITAL_OCEAN_API_KEY
        if not self.api_key:
            raise ValueError("Digital Ocean API key not provided")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.base_url = DIGITAL_OCEAN_API_BASE
    
    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make API request with error handling"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            
            if response.content:
                return response.json()
            return {}
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            raise
    
    def list_droplets(self, tag_name: Optional[str] = None) -> List[Dict]:
        """List all droplets, optionally filtered by tag"""
        endpoint = "droplets"
        if tag_name:
            endpoint += f"?tag_name={tag_name}"
        
        result = self._request("GET", endpoint)
        return result.get("droplets", [])
    
    def get_droplet(self, droplet_id: int) -> Dict:
        """Get details for a specific droplet"""
        return self._request("GET", f"droplets/{droplet_id}")["droplet"]
    
    def create_droplet(self, 
                      name: str,
                      size: str,
                      image: str,
                      region: str,
                      ssh_keys: List[str],
                      tags: Optional[List[str]] = None,
                      user_data: Optional[str] = None,
                      vpc_uuid: Optional[str] = None) -> Dict:
        """Create a new droplet"""
        data = {
            "name": name,
            "size": size,
            "image": image,
            "region": region,
            "ssh_keys": ssh_keys,
            "backups": False,
            "ipv6": True,
            "monitoring": True
        }
        
        if tags:
            data["tags"] = tags
        if user_data:
            data["user_data"] = user_data
        if vpc_uuid:
            data["vpc_uuid"] = vpc_uuid
        
        result = self._request("POST", "droplets", data)
        return result["droplet"]
    
    def delete_droplet(self, droplet_id: int) -> None:
        """Delete a droplet"""
        self._request("DELETE", f"droplets/{droplet_id}")
    
    def reboot_droplet(self, droplet_id: int) -> None:
        """Reboot a droplet"""
        self._request("POST", f"droplets/{droplet_id}/actions", {"type": "reboot"})
    
    def shutdown_droplet(self, droplet_id: int) -> None:
        """Shutdown a droplet"""
        self._request("POST", f"droplets/{droplet_id}/actions", {"type": "shutdown"})
    
    def power_on_droplet(self, droplet_id: int) -> None:
        """Power on a droplet"""
        self._request("POST", f"droplets/{droplet_id}/actions", {"type": "power_on"})
    
    def wait_for_droplet_status(self, droplet_id: int, status: str, timeout: int = 300) -> bool:
        """Wait for droplet to reach a specific status"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            droplet = self.get_droplet(droplet_id)
            if droplet["status"] == status:
                return True
            time.sleep(5)
        
        return False
    
    def list_ssh_keys(self) -> List[Dict]:
        """List all SSH keys in the account"""
        result = self._request("GET", "account/keys")
        return result.get("ssh_keys", [])
    
    def create_ssh_key(self, name: str, public_key: str) -> Dict:
        """Create a new SSH key"""
        data = {
            "name": name,
            "public_key": public_key
        }
        result = self._request("POST", "account/keys", data)
        return result["ssh_key"]
    
    def get_ssh_key_by_name(self, name: str) -> Optional[Dict]:
        """Get SSH key by name"""
        keys = self.list_ssh_keys()
        for key in keys:
            if key["name"] == name:
                return key
        return None
    
    def list_regions(self) -> List[Dict]:
        """List all available regions"""
        result = self._request("GET", "regions")
        return result.get("regions", [])
    
    def list_sizes(self) -> List[Dict]:
        """List all available droplet sizes"""
        result = self._request("GET", "sizes")
        return result.get("sizes", [])
    
    def list_images(self, type: Optional[str] = None) -> List[Dict]:
        """List available images"""
        endpoint = "images"
        if type:
            endpoint += f"?type={type}"
        
        result = self._request("GET", endpoint)
        return result.get("images", [])
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        result = self._request("GET", "account")
        return result.get("account", {})
    
    def list_volumes(self) -> List[Dict]:
        """List all volumes"""
        result = self._request("GET", "volumes")
        return result.get("volumes", [])
    
    def create_volume(self, name: str, size_gb: int, region: str, 
                     description: Optional[str] = None) -> Dict:
        """Create a new volume"""
        data = {
            "name": name,
            "size_gigabytes": size_gb,
            "region": region
        }
        if description:
            data["description"] = description
        
        result = self._request("POST", "volumes", data)
        return result["volume"]
    
    def attach_volume(self, volume_id: str, droplet_id: int) -> None:
        """Attach a volume to a droplet"""
        data = {
            "type": "attach",
            "droplet_id": droplet_id
        }
        self._request("POST", f"volumes/{volume_id}/actions", data)
    
    def detach_volume(self, volume_id: str, droplet_id: int) -> None:
        """Detach a volume from a droplet"""
        data = {
            "type": "detach",
            "droplet_id": droplet_id
        }
        self._request("POST", f"volumes/{volume_id}/actions", data)
    
    def delete_volume(self, volume_id: str) -> None:
        """Delete a volume"""
        self._request("DELETE", f"volumes/{volume_id}")
    
    def list_snapshots(self, resource_type: Optional[str] = None) -> List[Dict]:
        """List snapshots"""
        endpoint = "snapshots"
        if resource_type:
            endpoint += f"?resource_type={resource_type}"
        
        result = self._request("GET", endpoint)
        return result.get("snapshots", [])
    
    def create_droplet_snapshot(self, droplet_id: int, name: str) -> Dict:
        """Create a snapshot of a droplet"""
        data = {
            "type": "snapshot",
            "name": name
        }
        result = self._request("POST", f"droplets/{droplet_id}/actions", data)
        return result["action"]
    
    def get_droplet_neighbors(self, droplet_id: int) -> List[Dict]:
        """Get neighboring droplets on the same physical hardware"""
        result = self._request("GET", f"droplets/{droplet_id}/neighbors")
        return result.get("droplets", [])