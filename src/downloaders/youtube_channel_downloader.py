#!/usr/bin/env python3
"""
YouTube Channel Downloader with duplicate checking and background processing
"""
import subprocess
import json
import time
import threading
import queue
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib
import sys


class YouTubeChannelDownloader:
    """Download YouTube channel videos with proper state management"""
    
    def __init__(self, output_dir: str = "downloads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.state_file = self.output_dir / ".download_state.json"
        self.state = self._load_state()
        self.download_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.is_running = False
        self.download_thread = None
        
    def _load_state(self) -> Dict:
        """Load download state from file"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "downloaded": {},  # video_id: {title, file, date, size}
            "failed": {},      # video_id: {title, error, attempts}
            "channels": {}     # channel_url: {last_check, video_count}
        }
    
    def _save_state(self):
        """Save download state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _get_channel_videos(self, channel_url: str) -> List[Tuple[str, str, str]]:
        """Get list of videos from channel
        Returns: List of (video_id, title, url) tuples
        """
        print(f"Fetching video list from {channel_url}...")
        
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--print", '%(id)s|%(title)s|%(webpage_url)s|%(is_live)s|%(live_status)s',
            "--quiet",
            "--no-warnings",
            channel_url
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            videos = []
            
            for line in result.stdout.strip().split('\n'):
                if '|' in line:
                    parts = line.split('|', 4)
                    if len(parts) >= 3:
                        video_id, title, url = parts[:3]
                        
                        # Check live status
                        is_live = parts[3] if len(parts) > 3 else 'False'
                        live_status = parts[4] if len(parts) > 4 else 'not_live'
                        
                        # Skip if currently live streaming
                        if is_live == 'True' or live_status == 'is_live':
                            print(f"⚠️  Skipping live stream: {title}")
                            continue
                            
                        videos.append((video_id, title, url))
            
            # Update channel state
            self.state['channels'][channel_url] = {
                'last_check': datetime.now().isoformat(),
                'video_count': len(videos)
            }
            self._save_state()
            
            return videos
            
        except subprocess.CalledProcessError as e:
            print(f"Error fetching video list: {e}")
            return []
    
    def _is_downloaded(self, video_id: str) -> bool:
        """Check if video is already downloaded"""
        return video_id in self.state['downloaded']
    
    def _download_worker(self):
        """Background worker for downloading videos"""
        while self.is_running:
            try:
                # Get next video from queue (timeout to check is_running)
                video_info = self.download_queue.get(timeout=1)
                if video_info is None:  # Poison pill
                    break
                    
                video_id, title, url = video_info
                
                # Skip if already downloaded
                if self._is_downloaded(video_id):
                    print(f"Skipping {title} - already downloaded")
                    self.results_queue.put(('skipped', video_id, title))
                    continue
                
                # Download audio directly
                print(f"\n📥 Downloading audio: {title}")
                success, audio_path = self._download_audio_directly(video_id, title, url)
                
                if success:
                    self.results_queue.put(('success', video_id, title, audio_path))
                else:
                    self.results_queue.put(('failed', video_id, title))
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Download worker error: {e}")
    
    def _download_audio_directly(self, video_id: str, title: str, url: str) -> Tuple[bool, Optional[str]]:
        """Download audio directly without downloading full video"""
        # First check if it's a live stream using yt-dlp
        check_cmd = [
            "yt-dlp",
            "--print", "%(is_live)s|%(live_status)s",
            "--quiet",
            "--no-warnings",
            url
        ]
        
        try:
            result = subprocess.run(check_cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            if output:
                parts = output.split('|')
                is_live = parts[0] if len(parts) > 0 else 'False'
                live_status = parts[1] if len(parts) > 1 else 'not_live'
                
                if is_live == 'True' or live_status == 'is_live':
                    print(f"⚠️  Skipping live stream download: {title}")
                    return False, None
        except Exception as e:
            print(f"Warning: Could not check live status: {e}")
        
        # Clean filename
        clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title[:100]  # Limit length
        
        # Output path
        output_path = self.output_dir / f"{clean_title}_{video_id}.opus"
        
        cmd = [
            "yt-dlp",
            "-f", "ba[acodec=opus]/ba",  # Prefer opus, fallback to best audio
            "-x",  # Extract audio only
            "--audio-format", "opus",  # Keep as opus (no conversion if already opus)
            "--write-info-json",
            "--quiet",
            "--progress",
            "--no-warnings",
            "-o", str(output_path.with_suffix('.%(ext)s')),
            url
        ]
        
        try:
            # Check if failed too many times
            failed_attempts = self.state['failed'].get(video_id, {}).get('attempts', 0)
            if failed_attempts >= 3:
                print(f"Skipping {title} - failed {failed_attempts} times")
                return False, None
            
            # Download
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Success - update state
                duration = time.time() - start_time
                
                # Find the actual output file (yt-dlp might change the name)
                opus_files = list(self.output_dir.glob(f"{clean_title}_{video_id}*.opus"))
                if opus_files:
                    output_path = opus_files[0]
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    
                    # Verify reasonable size
                    if size_mb < 1:
                        print(f"✗ Audio file too small ({size_mb:.2f} MB), likely incomplete")
                        output_path.unlink()
                        return False, None
                else:
                    print(f"Warning: Could not find output file for {title}")
                    return False, None
                
                # Try to get upload date from info.json file
                upload_date = None
                info_json_path = output_path.with_suffix('.info.json')
                if info_json_path.exists():
                    try:
                        with open(info_json_path, 'r') as f:
                            info_data = json.load(f)
                            # yt-dlp stores upload date as YYYYMMDD string
                            if 'upload_date' in info_data and info_data['upload_date']:
                                date_str = info_data['upload_date']
                                upload_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                            elif 'release_date' in info_data and info_data['release_date']:
                                # Some videos use release_date instead
                                date_str = str(info_data['release_date'])
                                upload_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    except Exception as e:
                        print(f"Warning: Could not parse info.json: {e}")
                
                self.state['downloaded'][video_id] = {
                    'title': title,
                    'file': str(output_path),
                    'download_date': datetime.now().isoformat(),
                    'upload_date': upload_date,  # YouTube publication date
                    'size_mb': size_mb,
                    'download_time': duration
                }
                
                # Remove from failed if it was there
                if video_id in self.state['failed']:
                    del self.state['failed'][video_id]
                
                self._save_state()
                print(f"✓ Downloaded {title} ({size_mb:.1f} MB in {duration:.1f}s)")
                return True, str(output_path)
                
            else:
                # Failed - update state
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                
                if video_id not in self.state['failed']:
                    self.state['failed'][video_id] = {
                        'title': title,
                        'attempts': 0,
                        'errors': []
                    }
                
                self.state['failed'][video_id]['attempts'] += 1
                self.state['failed'][video_id]['errors'].append({
                    'date': datetime.now().isoformat(),
                    'error': error_msg
                })
                self.state['failed'][video_id]['last_error'] = error_msg
                
                self._save_state()
                print(f"✗ Failed to download {title}")
                return False, None
                
        except Exception as e:
            print(f"Error downloading {title}: {e}")
            return False, None
    
    def _download_video_only(self, video_id: str, title: str, url: str) -> Tuple[bool, Optional[str]]:
        """Download video without audio extraction"""
        # Clean filename
        clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title[:100]  # Limit length
        
        # Output path for video
        output_path = self.output_dir / f"{clean_title}_{video_id}.mp4"
        temp_path = self.output_dir / f"{clean_title}_{video_id}.mp4.part"
        
        # Check for existing partial download
        if temp_path.exists():
            print(f"⚠️  Removing partial download: {temp_path.name}")
            temp_path.unlink()
        
        cmd = [
            "yt-dlp",
            "-f", "best[ext=mp4]/best",  # Download best quality MP4
            "--write-info-json",
            "--quiet",
            "--progress",
            "--no-warnings",
            "-o", str(output_path),
            url
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and output_path.exists():
                duration = time.time() - start_time
                size_mb = output_path.stat().st_size / (1024 * 1024)
                
                # Verify file has reasonable size (at least 1MB)
                if size_mb < 1:
                    print(f"✗ Downloaded file too small ({size_mb:.2f} MB), likely incomplete")
                    output_path.unlink()
                    return False, None
                
                print(f"✓ Video downloaded: {size_mb:.1f} MB in {duration:.1f}s")
                return True, str(output_path)
            else:
                print(f"✗ Failed to download video: {title}")
                # Clean up any partial file
                if output_path.exists():
                    output_path.unlink()
                return False, None
                
        except Exception as e:
            print(f"Error downloading video {title}: {e}")
            # Clean up any partial file
            if output_path.exists():
                output_path.unlink()
            return False, None
    
    def _extract_audio(self, video_id: str, title: str, video_path: str) -> Tuple[bool, Optional[str]]:
        """Extract audio from video using ffmpeg"""
        # Clean filename
        clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title[:100]
        
        audio_path = self.output_dir / f"{clean_title}_{video_id}.mp3"
        
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "mp3",
            "-ab", "192k",
            "-ar", "44100",
            "-y",  # Overwrite
            str(audio_path)
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and audio_path.exists():
                duration = time.time() - start_time
                size_mb = audio_path.stat().st_size / (1024 * 1024)
                
                # Update state
                self.state['downloaded'][video_id] = {
                    'title': title,
                    'file': str(audio_path),
                    'date': datetime.now().isoformat(),
                    'size_mb': size_mb,
                    'extract_time': duration
                }
                
                # Remove from failed if it was there
                if video_id in self.state['failed']:
                    del self.state['failed'][video_id]
                
                self._save_state()
                print(f"✓ Audio extracted: {size_mb:.1f} MB in {duration:.1f}s")
                return True, str(audio_path)
            else:
                print(f"✗ Failed to extract audio: {title}")
                return False, None
                
        except Exception as e:
            print(f"Error extracting audio {title}: {e}")
            return False, None
    
    def _download_single_video(self, video_id: str, title: str, url: str) -> Tuple[bool, Optional[str]]:
        """Download a single video
        Returns: (success, filepath)
        """
        # Clean filename
        clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title[:100]  # Limit length
        
        # Output path
        output_path = self.output_dir / f"{clean_title}_{video_id}.mp3"
        
        cmd = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "--write-info-json",
            "--quiet",
            "--progress",
            "--no-warnings",
            "-o", str(output_path.with_suffix('.%(ext)s')),
            url
        ]
        
        try:
            # Check if failed too many times
            failed_attempts = self.state['failed'].get(video_id, {}).get('attempts', 0)
            if failed_attempts >= 3:
                print(f"Skipping {title} - failed {failed_attempts} times")
                return False, None
            
            # Download
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Success - update state
                duration = time.time() - start_time
                
                # Get file size
                if output_path.exists():
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                else:
                    # Check for actual output file (might have different name)
                    mp3_files = list(self.output_dir.glob(f"{clean_title}_{video_id}*.mp3"))
                    if mp3_files:
                        output_path = mp3_files[0]
                        size_mb = output_path.stat().st_size / (1024 * 1024)
                    else:
                        print(f"Warning: Could not find output file for {title}")
                        size_mb = 0
                
                self.state['downloaded'][video_id] = {
                    'title': title,
                    'file': str(output_path),
                    'date': datetime.now().isoformat(),
                    'size_mb': size_mb,
                    'download_time': duration
                }
                
                # Remove from failed if it was there
                if video_id in self.state['failed']:
                    del self.state['failed'][video_id]
                
                self._save_state()
                print(f"✓ Downloaded {title} ({size_mb:.1f} MB in {duration:.1f}s)")
                return True, str(output_path)
                
            else:
                # Failed - update state
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                
                if video_id not in self.state['failed']:
                    self.state['failed'][video_id] = {
                        'title': title,
                        'attempts': 0,
                        'errors': []
                    }
                
                self.state['failed'][video_id]['attempts'] += 1
                self.state['failed'][video_id]['errors'].append({
                    'date': datetime.now().isoformat(),
                    'error': error_msg
                })
                self.state['failed'][video_id]['last_error'] = error_msg
                
                self._save_state()
                print(f"✗ Failed to download {title}")
                return False, None
                
        except Exception as e:
            print(f"Error downloading {title}: {e}")
            return False, None
    
    def download_channel(self, channel_url: str, max_videos: Optional[int] = None) -> Dict:
        """Download all videos from a channel
        
        Args:
            channel_url: YouTube channel URL
            max_videos: Maximum number of videos to download
            
        Returns:
            Dictionary with download results
        """
        # Get video list
        videos = self._get_channel_videos(channel_url)
        
        if not videos:
            return {'error': 'No videos found'}
        
        print(f"Found {len(videos)} videos in channel")
        
        # Filter out already downloaded
        to_download = []
        for video_id, title, url in videos:
            if not self._is_downloaded(video_id):
                to_download.append((video_id, title, url))
        
        print(f"{len(to_download)} videos to download ({len(videos) - len(to_download)} already downloaded)")
        
        if max_videos and len(to_download) > max_videos:
            to_download = to_download[:max_videos]
            print(f"Limiting to {max_videos} videos")
        
        if not to_download:
            return {
                'total_videos': len(videos),
                'already_downloaded': len(videos),
                'new_downloads': 0
            }
        
        # Start download worker
        self.is_running = True
        self.download_thread = threading.Thread(target=self._download_worker, name="DownloadWorker")
        self.download_thread.start()
        
        # Queue all videos
        for video_info in to_download:
            self.download_queue.put(video_info)
        
        # Wait for downloads to complete
        print(f"\nStarting pipeline: {len(to_download)} videos to process...")
        print("Download → Extract Audio → Save\n")
        
        results = {
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'extract_failed': 0,
            'downloaded_files': []
        }
        
        # Process results as they come in
        completed = 0
        while completed < len(to_download):
            try:
                result = self.results_queue.get(timeout=1)
                completed += 1
                
                if result[0] == 'success':
                    results['success'] += 1
                    results['downloaded_files'].append(result[3])
                elif result[0] == 'failed':
                    results['failed'] += 1
                elif result[0] == 'skipped':
                    results['skipped'] += 1
                elif result[0] == 'extract_failed':
                    results['extract_failed'] += 1
                
                # Progress
                print(f"\nProgress: {completed}/{len(to_download)} completed "
                      f"(✓ {results['success']} ✗ {results['failed']} "
                      f"🎵 {results['extract_failed']} extract failed → {results['skipped']} skipped)")
                
            except queue.Empty:
                continue
        
        # Stop worker
        self.is_running = False
        self.download_queue.put(None)  # Poison pill
        self.download_thread.join()
        
        # Final results
        results['total_videos'] = len(videos)
        results['already_downloaded'] = len(videos) - len(to_download)
        results['new_downloads'] = results['success']
        
        print(f"\n=== Download Complete ===")
        print(f"Total videos in channel: {results['total_videos']}")
        print(f"Already downloaded: {results['already_downloaded']}")
        print(f"New downloads: {results['new_downloads']}")
        print(f"Failed: {results['failed']}")
        
        return results
    
    def get_downloaded_files(self) -> List[str]:
        """Get list of all downloaded files"""
        files = []
        # First try files from state
        for video_id, info in self.state['downloaded'].items():
            if Path(info['file']).exists():
                files.append(info['file'])
            # If not found at original path, check in output_dir
            elif (self.output_dir / Path(info['file']).name).exists():
                files.append(str(self.output_dir / Path(info['file']).name))
        
        # Also scan directory for audio files not in state
        for audio_file in self.output_dir.glob("*.opus"):
            if str(audio_file) not in files:
                files.append(str(audio_file))
        for audio_file in self.output_dir.glob("*.mp3"):
            if str(audio_file) not in files:
                files.append(str(audio_file))
                
        return sorted(files)
    
    def get_status(self) -> Dict:
        """Get current download status"""
        return {
            'downloaded_count': len(self.state['downloaded']),
            'failed_count': len(self.state['failed']),
            'total_size_mb': sum(
                info.get('size_mb', 0) 
                for info in self.state['downloaded'].values()
            ),
            'channels': self.state['channels']
        }
    
    def retry_failed(self):
        """Reset failed downloads for retry"""
        print(f"Resetting {len(self.state['failed'])} failed downloads for retry")
        self.state['failed'] = {}
        self._save_state()
    
    def verify_downloads(self) -> Dict[str, List[str]]:
        """Verify all downloaded files exist and are valid
        
        Returns:
            Dict with 'valid', 'missing', and 'corrupted' lists
        """
        results = {
            'valid': [],
            'missing': [],
            'corrupted': []
        }
        
        for video_id, info in self.state['downloaded'].items():
            filepath = Path(info['file'])
            
            if not filepath.exists():
                results['missing'].append(video_id)
                print(f"⚠️  Missing file for {info['title']}: {filepath}")
            elif filepath.stat().st_size < 1_000_000:  # Less than 1MB
                results['corrupted'].append(video_id)
                print(f"⚠️  File too small for {info['title']}: {filepath.stat().st_size / 1_000_000:.2f} MB")
            else:
                results['valid'].append(video_id)
        
        # Clean up state for missing/corrupted files
        if results['missing'] or results['corrupted']:
            print("\nCleaning up state for invalid files...")
            for video_id in results['missing'] + results['corrupted']:
                del self.state['downloaded'][video_id]
            self._save_state()
            print(f"Removed {len(results['missing']) + len(results['corrupted'])} entries from state")
        
        return results


if __name__ == "__main__":
    # Example usage
    downloader = YouTubeChannelDownloader("bitcoin_dive_bar_downloads")
    
    # Download Bitcoin Dive Bar channel
    results = downloader.download_channel(
        "https://www.youtube.com/@BitcoinDiveBar",
        max_videos=None  # Download all
    )
    
    # Show downloaded files
    print("\nDownloaded files:")
    for filepath in downloader.get_downloaded_files():
        print(f"  {filepath}")