#!/usr/bin/env python3
"""
Show only the latest run's predictions
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import calendar

def convert_timeframe_to_concrete(timeframe, time_value, episode_date, predicted_date=None, raw_timeframe=None):
    """Convert relative timeframes to concrete dates based on episode date"""
    if not episode_date:
        return timeframe
    
    # If timeframe is None (Stage 3 hasn't parsed it yet), show the raw timeframe
    if timeframe is None or timeframe == 'None':
        if raw_timeframe:
            return f"{raw_timeframe} [unparsed]"
        return "No timeframe specified"
    
    try:
        # Parse episode date
        ep_date = datetime.strptime(episode_date, '%Y-%m-%d')
        
        # If we already have a predicted_date, use it
        if predicted_date:
            return predicted_date
        
        # Convert based on timeframe type
        if timeframe == 'days':
            days = time_value if time_value is not None else 1
            target_date = ep_date + timedelta(days=days)
            return f"{target_date.strftime('%B %d, %Y')} (~{days} days from episode)"
            
        elif timeframe == 'weeks':
            weeks = time_value if time_value is not None else 1
            target_date = ep_date + timedelta(weeks=weeks)
            # Determine which week of the month
            week_of_month = (target_date.day - 1) // 7 + 1
            ordinal = ['1st', '2nd', '3rd', '4th', '5th'][min(week_of_month - 1, 4)]
            return f"{ordinal} week of {target_date.strftime('%B %Y')} (~{weeks} week{'s' if weeks > 1 else ''} from episode)"
            
        elif timeframe == 'months':
            months = time_value if time_value is not None else 1
            # Add months (approximate)
            target_month = ep_date.month + months
            target_year = ep_date.year
            while target_month > 12:
                target_month -= 12
                target_year += 1
            return f"{calendar.month_name[target_month]} {target_year} (~{months} month{'s' if months > 1 else ''} from episode)"
            
        elif timeframe == 'end_of_year':
            return f"End of {ep_date.year}"
            
        elif timeframe == 'specific_date' and predicted_date:
            pred_date = datetime.strptime(predicted_date, '%Y-%m-%d')
            return pred_date.strftime('%B %d, %Y')
            
        elif timeframe == 'None' or not timeframe:
            return 'No timeframe specified'
            
        else:
            # For other timeframes, just return as is
            return timeframe
            
    except Exception as e:
        # If parsing fails, return original timeframe
        return timeframe

# Load predictions
pred_file = Path('data/episodes/bitcoin_dive_bar_analysis/prediction_data/latest/predictions.json')
if not pred_file.exists():
    print(f"No predictions file found at {pred_file}")
    exit(1)

with open(pred_file) as f:
    data = json.load(f)

# Get the latest run ID
latest_run_id = data.get('latest_run_id', 'unknown')
latest_model = data.get('latest_model', 'unknown')

# Filter predictions for the latest run only
all_predictions = data.get('all_predictions', [])
latest_predictions = [p for p in all_predictions if p.get('run_id') == latest_run_id]

print(f"LATEST RUN: {latest_run_id}")
print(f"Model: {latest_model}")
print(f"Predictions in this run: {len(latest_predictions)}")
print("="*80)

# Group by podcast series first, then episode
by_series = defaultdict(lambda: defaultdict(list))
for pred in latest_predictions:
    series = pred.get('podcast_series', 'Unknown Podcast')
    episode = pred.get('episode', 'Unknown Episode')
    by_series[series][episode].append(pred)

for series, episodes in sorted(by_series.items()):
    print(f"\n📻 {series}:")
    for episode, episode_preds in sorted(episodes.items()):
        # Shorten episode name by removing series prefix if present
        display_episode = episode
        if episode.startswith(series):
            display_episode = episode[len(series):].strip(' -')
        print(f"\n  {display_episode} ({len(episode_preds)} predictions):")
        
        for i, pred in enumerate(episode_preds, 1):
            asset = pred.get('asset', 'Unknown')
            value = pred.get('value', 0)
            confidence = pred.get('confidence', 'unknown')
            
            # Get timeframe and convert to concrete date
            raw_timeframe = pred.get('timeframe')
            time_value = pred.get('time_value')
            episode_date = pred.get('episode_date')
            predicted_date = pred.get('predicted_date')
            
            # Get raw timeframe from parsing info if available
            raw_timeframe_str = None
            parsing_info = pred.get('timeframe_parsing_info', {})
            if parsing_info:
                raw_timeframe_str = parsing_info.get('original_timeframe')
            
            # Convert relative timeframe to concrete date
            concrete_timeframe = convert_timeframe_to_concrete(
                raw_timeframe, time_value, episode_date, predicted_date, raw_timeframe_str
            )
            
            # Format price
            if value >= 1000:
                price_str = f'${value:,.0f}'
            else:
                price_str = f'${value}'
            
            # Get YouTube link
            timestamp = pred.get('timestamp')
            episode_id = pred.get('episode_id')
            youtube_link = None
            
            # Check if we have a direct youtube_link field
            if pred.get('youtube_link'):
                youtube_link = pred['youtube_link']
            # Otherwise, try to construct it
            elif episode_id and timestamp:
                # Parse timestamp to seconds
                try:
                    # Skip invalid timestamps
                    if timestamp in ['unknown', '00:00:00', 'None'] or not timestamp:
                        youtube_link = f"https://youtube.com/watch?v={episode_id}"
                    else:
                        parts = timestamp.split(':')
                        if len(parts) == 2:  # MM:SS
                            seconds = int(parts[0]) * 60 + int(parts[1])
                        elif len(parts) == 3:  # H:MM:SS
                            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                        else:
                            seconds = 0
                        
                        # Only add timestamp if it's greater than 0
                        if seconds > 0:
                            youtube_link = f"https://youtube.com/watch?v={episode_id}&t={seconds}s"
                        else:
                            youtube_link = f"https://youtube.com/watch?v={episode_id}"
                except:
                    youtube_link = f"https://youtube.com/watch?v={episode_id}"
            
            print(f"    {i}. {asset} → {price_str}")
            print(f"       Confidence: {confidence}")
            print(f"       Timeframe: {concrete_timeframe}")
            if episode_date:
                print(f"       Episode Date: {episode_date}")
            
            # Show timeframe parsing debug info
            parsing_info = pred.get('timeframe_parsing_info', {})
            if parsing_info:
                print(f"       📊 Parsing Flow: {parsing_info.get('parsing_flow', 'unknown')}")
                if parsing_info.get('original_timeframe'):
                    print(f"       Original: \"{parsing_info['original_timeframe']}\"")
                if parsing_info.get('parsing_flow') == 'deferred_to_stage3':
                    print(f"       Status: Awaiting Stage 3 parsing")
                else:
                    if parsing_info.get('timeframe_type'):
                        print(f"       Type: {parsing_info['timeframe_type']}")
                    if parsing_info.get('timeframe_value') is not None:
                        print(f"       Value: {parsing_info['timeframe_value']}")
                if parsing_info.get('parsing_notes'):
                    for note in parsing_info['parsing_notes']:
                        print(f"       Note: {note}")
            
            if youtube_link:
                print(f"       YouTube: {youtube_link}")
            elif timestamp:
                print(f"       Timestamp: {timestamp} (no video ID)")
            else:
                print(f"       ⚠️  Missing timestamp")
            if pred.get('reasoning'):
                print(f"       Reasoning: {pred['reasoning'][:100]}...")
            if pred.get('context'):
                print(f"       Context: \"{pred['context'][:80]}...\"")

print(f"\n{'='*80}")
print(f"Total predictions in latest run: {len(latest_predictions)}")