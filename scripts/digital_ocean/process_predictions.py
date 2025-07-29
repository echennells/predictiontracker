#!/usr/bin/env python3
"""
Process predictions from transcript on Digital Ocean droplet
This runs on the DO droplet, not locally
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# Add paths for imports
sys.path.insert(0, '/workspace/xtotext')
sys.path.insert(0, '/workspace/xtotext/src')
sys.path.insert(0, '/workspace/xtotext/config')

from predictions.prediction_tracker.tracker_two_stage import TwoStageCryptoPredictionTracker


def main():
    # Import config to show model being used
    import config
    import os
    
    # Use batch run ID if provided, otherwise generate unique run ID for this extraction
    run_id = os.environ.get('BATCH_RUN_ID', datetime.now().strftime('%Y%m%d_%H%M%S'))
    print(f"Starting prediction extraction - Run ID: {run_id}")
    print(f"Using LLM model: {config.LLM_MODEL}")
    
    # Load transcript
    transcript_path = Path('/workspace/transcript.json')
    if not transcript_path.exists():
        print(f"Error: Transcript not found at {transcript_path}")
        sys.exit(1)
    
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    # Load episode info if available
    episode_info_path = Path('/workspace/episode_info.json')
    if episode_info_path.exists():
        with open(episode_info_path, 'r') as f:
            episode_info = json.load(f)
            # Use upload_date if available, otherwise fall back to today
            episode_info['date'] = episode_info.get('upload_date') or datetime.now().strftime('%Y-%m-%d')
            episode_info['id'] = episode_info.get('video_id', 'unknown-id')
    else:
        # Fallback to extracting from transcript
        episode_info = {
            'title': transcript_data.get('title', 'Unknown Episode'),
            'date': transcript_data.get('upload_date', transcript_data.get('date', datetime.now().strftime('%Y-%m-%d'))),
            'id': transcript_data.get('id', 'unknown-id'),
            'video_id': transcript_data.get('video_id')
        }
    
    print(f"Processing predictions for: {episode_info['title']}")
    
    # Load existing predictions if they exist
    latest_path = Path('/workspace/predictions.json')
    existing_predictions = []
    if latest_path.exists():
        try:
            with open(latest_path, 'r') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, dict) and 'all_predictions' in existing_data:
                    existing_predictions = existing_data['all_predictions']
                print(f"Loaded {len(existing_predictions)} existing predictions")
        except Exception as e:
            print(f"Error loading existing predictions: {e}")
    
    # Process predictions using two-stage approach
    tracker = TwoStageCryptoPredictionTracker()
    predictions = tracker.process_episode(str(transcript_path), episode_info)
    
    # Extract podcast series name from episode title
    episode_title = episode_info['title']
    podcast_series = 'Unknown Podcast'
    
    # Try to extract series name (usually before "EP" or "Episode")
    if ' EP ' in episode_title:
        podcast_series = episode_title.split(' EP ')[0].strip()
    elif ' Episode ' in episode_title:
        podcast_series = episode_title.split(' Episode ')[0].strip()
    elif ' - ' in episode_title:
        # Some podcasts use " - " as separator
        podcast_series = episode_title.split(' - ')[0].strip()
    
    print(f"Podcast series: {podcast_series}")
    
    # Convert predictions to serializable format with run_id
    new_predictions = []
    for p in predictions:
        try:
            # Format value with error handling
            try:
                value_text = f"${float(p.value):,.0f}"
            except Exception as e:
                print(f"Error formatting value {p.value}: {e}")
                value_text = f"${p.value}"
            
            pred_dict = {
                'text': f"{p.asset} to {value_text}",
                'asset': p.asset,
                'value': float(p.value),
                'confidence': p.confidence.value if hasattr(p.confidence, 'value') else str(p.confidence),
                'timestamp': p.timestamp,
                'context': p.raw_text[:200] if p.raw_text else '',
                'timeframe': p.time_frame.value if p.time_frame and hasattr(p.time_frame, 'value') else str(p.time_frame),
                'predicted_date': p.predicted_date,
                'time_value': p.time_value,
                'reasoning': p.reasoning,
                'podcast_series': p.podcast_series or podcast_series,  # Use extracted series name
                'episode': episode_info['title'],
                'episode_id': episode_info['id'],
                'episode_date': p.episode_date,  # YouTube upload date
                'extraction_date': datetime.now().isoformat(),
                'run_id': run_id,  # Add run_id to track this extraction run
                'model_used': config.LLM_MODEL,  # Track which model was used
                'prediction_id': p.prediction_id,
                'timeframe_parsing_info': p.timeframe_parsing_info  # Add debug info
            }
            new_predictions.append(pred_dict)
        except Exception as e:
            print(f"Error processing prediction: {e}")
            continue
    
    # Combine with existing predictions
    all_predictions = existing_predictions + new_predictions
    
    # Create results structure
    results = {
        'latest_run_id': run_id,
        'latest_model': config.LLM_MODEL,
        'latest_episode': episode_info['title'],
        'latest_episode_id': episode_info['id'],
        'latest_predictions_count': len(new_predictions),
        'total_predictions_count': len(all_predictions),
        'runs': sorted(list(set(p.get('run_id', 'unknown') for p in all_predictions))),
        'all_predictions': all_predictions
    }
    
    # Save versioned file for this run
    run_output_path = Path(f'/workspace/predictions_{run_id}.json')
    run_results = {
        'run_id': run_id,
        'episode': episode_info['title'],
        'episode_id': episode_info['id'],
        'predictions': new_predictions
    }
    with open(run_output_path, 'w') as f:
        json.dump(run_results, f, indent=2)
    
    # Save cumulative results
    with open(latest_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRun {run_id} completed:")
    print(f"  - Found {len(new_predictions)} new predictions")
    print(f"  - Total predictions across all runs: {len(all_predictions)}")
    print(f"  - Results saved to {run_output_path}")
    print(f"  - Cumulative results updated in {latest_path}")
    
    # Copy debug logs if they exist
    debug_dir = Path('/workspace/logs/debug')
    if debug_dir.exists():
        debug_files = list(debug_dir.glob('debug_*.json'))
        if debug_files:
            print(f"\nFound {len(debug_files)} debug log files")
            # Copy them to a location that will be retrieved
            for debug_file in debug_files:
                dest = Path('/workspace') / debug_file.name
                import shutil
                shutil.copy2(debug_file, dest)
                print(f"  - Copied {debug_file.name} to /workspace/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())