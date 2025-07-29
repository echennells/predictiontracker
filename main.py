#!/usr/bin/env python3
"""
Main entry point for xtotext Bitcoin prediction extraction

Flow:
1. Download YouTube files locally
2. Upload to Digital Ocean droplet
3. Run processing on DO (which calls Vast.ai for GPU transcription)
4. Download results back to local
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from infrastructure.digital_ocean.simple_runner import SimpleDigitalOceanRunner
from downloaders.youtube_channel_downloader import YouTubeChannelDownloader


def main():
    # Set up logging
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"prediction_extraction_{log_timestamp}.log"
    
    # Configure logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting prediction extraction run - Log file: {log_file}")
    
    # Configuration
    channel_url = "https://www.youtube.com/@BitcoinDiveBar"
    local_download_dir = Path("data/episodes/bitcoin_dive_bar_downloads")
    output_dir = Path("data/episodes/bitcoin_dive_bar_analysis")
    
    local_download_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download YouTube files locally
    print("="*80)
    print("STEP 1: Downloading YouTube files locally")
    print("="*80)
    logger.info("="*80)
    logger.info("STEP 1: Downloading YouTube files locally")
    logger.info("="*80)
    
    downloader = YouTubeChannelDownloader(str(local_download_dir))
    
    # Always check for new episodes
    print(f"Checking {channel_url} for new episodes...")
    download_result = downloader.download_channel(channel_url)  # No max_videos limit
    
    if 'new_downloads' in download_result:
        print(f"Downloaded {download_result['new_downloads']} new episodes")
    
    audio_files = [Path(f) for f in downloader.get_downloaded_files() if Path(f).exists()]
    print(f"Found {len(audio_files)} total episodes locally")
    
    if not audio_files:
        print("No episodes found to process")
        return
    
    # Get Vast.ai API key for DO droplet to use
    vast_api_key = os.getenv("VAST_API_KEY")
    if not vast_api_key:
        api_key_file = Path.home() / ".config/vastai/vast_api_key"
        if api_key_file.exists():
            vast_api_key = api_key_file.read_text().strip()
    
    if not vast_api_key:
        print("Error: No Vast.ai API key found (needed for GPU transcription)")
        return
    
    # Process all episodes
    print(f"\nWill process {len(audio_files)} episodes")
    
    # Create timestamp for this run
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Results tracking
    all_results = []
    
    # Process each episode
    for i, audio_file in enumerate(audio_files):
        print("\n" + "="*80)
        print(f"EPISODE {i+1}/{len(audio_files)}: {audio_file.name}")
        print("="*80)
        
        # Check if transcript already exists
        transcripts_dir = output_dir / "transcripts"
        transcripts_dir.mkdir(exist_ok=True)
        transcript_file = transcripts_dir / f"{audio_file.stem}_transcript.json"
        
        if transcript_file.exists():
            print(f"Transcript already exists, skipping Vast.ai transcription")
        else:
            # Step 2: Process audio on Vast.ai
            print("\nTranscribing with Vast.ai GPU...")
            
            from infrastructure.vast_ai.transcription_runner import TranscriptionRunner
            
            vast_runner = TranscriptionRunner(vast_api_key)
            
            try:
                # Set up GPU instance
                print("Starting Vast.ai GPU instance...")
                instance = vast_runner.setup_instance(
                    gpu_type="RTX 3080",
                    max_price=0.30
                )
                print(f"GPU instance {instance['id']} ready")
                
                # Transcribe
                transcript_result = vast_runner.transcribe_audio(
                    audio_path=audio_file,
                    output_dir=transcripts_dir,
                    model="base",
                    use_faster_whisper=True
                )
                
                print(f"Transcription completed in {transcript_result['metadata']['transcription_time']:.1f}s")
                
            finally:
                # Cleanup GPU immediately after use
                print("Cleaning up Vast.ai GPU instance...")
                vast_runner.cleanup(destroy_instance=True)
    
        # After processing this episode, track result
        episode_result = {
            'episode': audio_file.name,
            'transcript': transcript_file.exists(),
            'predictions': False,
            'error': None
        }
        all_results.append(episode_result)
    
    # Step 3: Process all transcripts on Digital Ocean
    print("\n" + "="*80)
    print("STEP 3: Extracting predictions on Digital Ocean")
    print("="*80)
    
    # Generate a single run_id for this entire batch
    batch_run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Batch run ID: {batch_run_id}")
    
    # Create runner without context manager so we can control cleanup
    runner = SimpleDigitalOceanRunner()
    cleanup_on_exit = True  # Flag to control cleanup
    
    try:
        # Start DO droplet
        runner.start(droplet_size="s-2vcpu-4gb", wait_time=60)
        
        # Upload code and scripts
        print("\nUploading code to Digital Ocean...")
        runner.upload_code(Path("src"))
        runner.upload_code(Path("config"))
        runner.upload_code(Path("scripts"))
        
        # Setup environment once
        print("\nSetting up Python environment...")
        runner.run_command("chmod +x /workspace/xtotext/scripts/digital_ocean/setup_environment.sh")
        setup_result = runner.run_command("bash /workspace/xtotext/scripts/digital_ocean/setup_environment.sh")
        
        if not setup_result['success']:
            print(f"Warning: Setup had issues: {setup_result['stderr']}")
        
        # Process each episode that has a transcript
        for i, result in enumerate(all_results):
            if not result['transcript']:
                continue
                
            audio_file = Path(local_download_dir) / result['episode']
            transcript_file = transcripts_dir / f"{audio_file.stem}_transcript.json"
            
            print(f"\nProcessing predictions for episode {i+1}: {audio_file.stem}")
            
            # Upload transcript
            runner.ssh.upload_file(transcript_file, "/workspace/transcript.json")
            
            # Extract video ID from filename
            # YouTube video IDs are always 11 characters and appear at the end
            video_id = None
            stem = audio_file.stem
            if len(stem) >= 11 and '_' in stem:
                # Extract the last 11 characters after the last occurrence of '_' 
                # that leaves at least 11 characters
                for i in range(len(stem) - 11, -1, -1):
                    if stem[i] == '_' and len(stem) - i - 1 >= 11:
                        potential_id = stem[i+1:i+12]
                        # Validate it looks like a YouTube ID (11 chars, alphanumeric + _ and -)
                        if len(potential_id) == 11 and all(c.isalnum() or c in '_-' for c in potential_id):
                            video_id = potential_id
                            break
            
            # Get upload date from downloader state if available
            upload_date = None
            if video_id and downloader.state.get('downloaded', {}).get(video_id):
                upload_date = downloader.state['downloaded'][video_id].get('upload_date')
            
            # Create episode info file
            episode_info = {
                'title': audio_file.stem.replace('_', ' '),
                'video_id': video_id,
                'filename': audio_file.name,
                'upload_date': upload_date  # YouTube publication date
            }
            
            # Upload episode info
            runner.ssh.execute_command(f"echo '{json.dumps(episode_info)}' > /workspace/episode_info.json")
            
            # Extract predictions with batch run_id
            extract_cmd = f"""
cd /workspace
source venv/bin/activate
pip install tiktoken>=0.5.0
BATCH_RUN_ID={batch_run_id} python /workspace/xtotext/scripts/digital_ocean/process_predictions.py
"""
            
            pred_result = runner.run_command(extract_cmd, timeout=300)
            
            if pred_result['success']:
                print("Prediction extraction completed!")
                logger.info(f"Prediction extraction completed for {audio_file.stem}")
                # Print stdout to see debugging output
                if pred_result.get('stdout'):
                    print("\n--- PREDICTION EXTRACTION OUTPUT ---")
                    print(pred_result['stdout'][:5000])  # First 5000 chars
                    print("--- END OUTPUT ---\n")
                    # Also log the full output to file
                    logger.info("--- PREDICTION EXTRACTION OUTPUT ---")
                    logger.info(pred_result['stdout'])  # Log full output
                    logger.info("--- END OUTPUT ---")
                
                # Download results
                predictions_dir = output_dir / "predictions"
                predictions_dir.mkdir(exist_ok=True)
                
                # Create versioned directory for this run
                version_dir = predictions_dir / f"run_{run_timestamp}"
                version_dir.mkdir(exist_ok=True)
                
                # Download the versioned prediction file
                runner.ssh.download_file(
                    "/workspace/predictions.json",
                    version_dir / f"{audio_file.stem}_predictions.json"
                )
                
                # Also save to the main predictions directory
                runner.ssh.download_file(
                    "/workspace/predictions.json",
                    predictions_dir / f"{audio_file.stem}_predictions.json"
                )
                
                result['predictions'] = True
            else:
                print(f"Prediction extraction failed: {pred_result['stderr']}")
                logger.error(f"Prediction extraction failed for {audio_file.stem}: {pred_result['stderr']}")
                result['error'] = pred_result['stderr']
                # Also print stdout in case of failure
                if pred_result.get('stdout'):
                    print("\n--- FAILED EXTRACTION OUTPUT ---")
                    print(pred_result['stdout'][:5000])
                    print("--- END OUTPUT ---\n")
                    # Log full output
                    logger.error("--- FAILED EXTRACTION OUTPUT ---")
                    logger.error(pred_result['stdout'])
                    logger.error("--- END OUTPUT ---")
            
        
        # Sync the full prediction database after all episodes
        print("\n" + "="*80)
        print("Syncing prediction database from Digital Ocean...")
        print("="*80)
        
        # Create versioned database directory
        db_version_dir = output_dir / "prediction_data" / f"version_{run_timestamp}"
        db_version_dir.mkdir(parents=True, exist_ok=True)
        
        # Also keep a "latest" directory
        local_db_dir = output_dir / "prediction_data" / "latest"
        local_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Download all database files
        db_files = [
            "episodes.json", 
            "speakers.json",
            "outcomes.json"
        ]
        
        # First download the cumulative predictions.json from our new location
        if runner.ssh.file_exists("/workspace/predictions.json"):
            print("Downloading cumulative predictions from /workspace/predictions.json")
            runner.ssh.download_file(
                "/workspace/predictions.json",
                local_db_dir / "predictions.json"
            )
            runner.ssh.download_file(
                "/workspace/predictions.json", 
                db_version_dir / "predictions.json"
            )
        
        # Then download other database files from the old location
        for db_file in db_files:
            remote_path = f"/workspace/prediction_data/{db_file}"
            
            # Check if remote file exists
            if runner.ssh.file_exists(remote_path):
                print(f"  Downloading {db_file}...")
                # Download to versioned directory
                runner.ssh.download_file(remote_path, db_version_dir / db_file)
                # Also download to latest directory
                runner.ssh.download_file(remote_path, local_db_dir / db_file)
            else:
                print(f"  {db_file} not found on remote (might not be created yet)")
        
        print(f"Database synced to:")
        print(f"  - Version: {db_version_dir}")
        print(f"  - Latest: {local_db_dir}")
        
        # Download debug logs if they exist
        print("\nChecking for debug logs...")
        exit_code, stdout, stderr = runner.ssh.execute_command("ls /workspace/debug_*.json 2>/dev/null || true")
        debug_files = stdout.strip().split('\n') if stdout.strip() else []
        debug_files = [f for f in debug_files if f and 'debug_' in f]
        
        if debug_files:
            debug_dir = output_dir / "logs" / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"Found {len(debug_files)} debug log files")
            
            for debug_file in debug_files:
                if debug_file.strip():
                    filename = debug_file.split('/')[-1]
                    print(f"  Downloading {filename}...")
                    runner.ssh.download_file(debug_file, debug_dir / filename)
            
            print(f"Debug logs downloaded to: {debug_dir}")
            
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("\n" + "="*80)
        print("KEEPING DROPLET ALIVE FOR DEBUGGING")
        print("="*80)
        print(f"SSH into droplet: ssh root@{runner.droplet_ip}")
        print(f"Droplet ID: {runner.droplet.get('id') if runner.droplet else 'Unknown'}")
        print("\nTo destroy manually, run:")
        print(f"  python3 -c \"from src.infrastructure.digital_ocean.simple_runner import SimpleDigitalOceanRunner; r = SimpleDigitalOceanRunner(); r.cleanup()\"")
        print("="*80)
        cleanup_on_exit = False  # Don't cleanup on error
        raise  # Re-raise the exception
    
    finally:
        # Clean up Digital Ocean droplet if flag is set
        if cleanup_on_exit and 'runner' in locals() and runner.droplet:
            print("\n" + "="*80)
            print("Cleaning up Digital Ocean droplet...")
            print("="*80)
            try:
                runner.cleanup()
                print("✓ Droplet destroyed successfully")
            except Exception as cleanup_error:
                print(f"✗ Error cleaning up droplet: {cleanup_error}")
                print(f"  Droplet ID: {runner.droplet.get('id') if runner.droplet else 'Unknown'}")
                print("  You may need to manually destroy it in the Digital Ocean console")
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    
    # Show results for each episode
    print("\nEpisode Results:")
    successful = 0
    for result in all_results:
        status = "✓" if result['predictions'] else "✗"
        print(f"  {status} {result['episode']}")
        if result['predictions']:
            successful += 1
        elif result['error']:
            print(f"    Error: {result['error'][:100]}...")
    
    print(f"\nSummary:")
    print(f"  Total episodes: {len(all_results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(all_results) - successful}")
    print(f"  YouTube files: {local_download_dir}")
    print(f"  Results: {output_dir}")
    
    logger.info("="*80)
    logger.info(f"Processing complete. Total: {len(all_results)}, Successful: {successful}, Failed: {len(all_results) - successful}")
    logger.info(f"Log file saved to: {log_file}")
    
    print(f"\n📝 Full log saved to: {log_file}")


if __name__ == "__main__":
    main()
