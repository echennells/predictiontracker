"""
Storage system for predictions and outcomes
"""
import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from .models import Prediction, Outcome, Speaker

class PredictionStorage:
    def __init__(self, data_dir: str = "prediction_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # File paths
        self.predictions_file = self.data_dir / "predictions.json"
        self.outcomes_file = self.data_dir / "outcomes.json"
        self.speakers_file = self.data_dir / "speakers.json"
        self.episodes_file = self.data_dir / "episodes.json"
        
        # Initialize files if they don't exist
        for file_path in [self.predictions_file, self.outcomes_file, 
                         self.speakers_file, self.episodes_file]:
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    json.dump([], f)
    
    def save_predictions(self, predictions: List[Prediction], merge: bool = True):
        """Save predictions to storage"""
        if merge:
            existing = self.load_predictions()
            # Only add new predictions
            existing_ids = {p.prediction_id for p in existing}
            new_predictions = [p for p in predictions if p.prediction_id not in existing_ids]
            predictions = existing + new_predictions
        
        data = [p.to_dict() for p in predictions]
        
        # Custom JSON encoder to handle Decimal types
        from decimal import Decimal
        
        def decimal_default(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
        
        with open(self.predictions_file, 'w') as f:
            json.dump(data, f, indent=2, default=decimal_default)
    
    def load_predictions(self) -> List[Prediction]:
        """Load all predictions"""
        with open(self.predictions_file, 'r') as f:
            data = json.load(f)
        return [Prediction.from_dict(d) for d in data]
    
    def save_outcome(self, outcome: Outcome):
        """Save a single outcome"""
        outcomes = self.load_outcomes()
        
        # Check if outcome already exists
        existing = [o for o in outcomes if o.prediction_id != outcome.prediction_id]
        existing.append(outcome)
        
        data = [o.to_dict() for o in existing]
        
        # Custom JSON encoder to handle Decimal types
        from decimal import Decimal
        
        def decimal_default(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
        
        with open(self.outcomes_file, 'w') as f:
            json.dump(data, f, indent=2, default=decimal_default)
    
    def load_outcomes(self) -> List[Outcome]:
        """Load all outcomes"""
        with open(self.outcomes_file, 'r') as f:
            data = json.load(f)
        return [Outcome(**d) for d in data]
    
    def get_predictions_by_asset(self, asset: str) -> List[Prediction]:
        """Get all predictions for a specific asset"""
        predictions = self.load_predictions()
        return [p for p in predictions if p.asset == asset]
    
    def get_predictions_by_date_range(self, start_date: str, end_date: str) -> List[Prediction]:
        """Get predictions within a date range"""
        predictions = self.load_predictions()
        return [p for p in predictions 
                if start_date <= p.episode_date <= end_date]
    
    def get_pending_predictions(self) -> List[Prediction]:
        """Get predictions without outcomes"""
        predictions = self.load_predictions()
        outcomes = self.load_outcomes()
        outcome_ids = {o.prediction_id for o in outcomes}
        
        return [p for p in predictions if p.prediction_id not in outcome_ids]
    
    def mark_episode_processed(self, episode_id: str, episode_data: Dict):
        """Track which episodes have been processed"""
        with open(self.episodes_file, 'r') as f:
            episodes = json.load(f)
        
        # Add or update episode
        episode_data['id'] = episode_id
        episode_data['processed_date'] = datetime.now().isoformat()
        
        # Remove old entry if exists
        episodes = [e for e in episodes if e.get('id') != episode_id]
        episodes.append(episode_data)
        
        with open(self.episodes_file, 'w') as f:
            json.dump(episodes, f, indent=2)
    
    def is_episode_processed(self, episode_id: str) -> bool:
        """Check if episode has been processed"""
        with open(self.episodes_file, 'r') as f:
            episodes = json.load(f)
        return any(e.get('id') == episode_id for e in episodes)
    
    def update_speaker_stats(self):
        """Update all speaker statistics"""
        predictions = self.load_predictions()
        outcomes = self.load_outcomes()
        
        # Group by speaker
        speakers_dict = {}
        for pred in predictions:
            if pred.speaker:
                if pred.speaker not in speakers_dict:
                    speakers_dict[pred.speaker] = Speaker(name=pred.speaker)
        
        # Update stats for each speaker
        speakers = []
        for speaker in speakers_dict.values():
            speaker.update_stats(predictions, outcomes)
            speakers.append(speaker)
        
        # Save
        data = [vars(s) for s in speakers]
        with open(self.speakers_file, 'w') as f:
            json.dump(data, f, indent=2)