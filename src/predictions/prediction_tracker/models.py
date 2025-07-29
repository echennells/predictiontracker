"""
Data models for crypto prediction tracking
"""
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, field, asdict
from enum import Enum

class PredictionType(Enum):
    PRICE_TARGET = "price_target"
    PERCENTAGE_CHANGE = "percentage_change"
    MARKET_CAP = "market_cap"
    RANKING = "ranking"
    EVENT = "event"

class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"

class TimeFrame(Enum):
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"
    SPECIFIC_DATE = "specific_date"
    EOY = "end_of_year"
    EOQ = "end_of_quarter"
    EOM = "end_of_month"

@dataclass
class Prediction:
    """A single price prediction"""
    # Core prediction data
    asset: str  # BTC, ETH, SMLR, etc.
    prediction_type: PredictionType
    value: float  # Price in USD or percentage
    
    # Timing
    predicted_date: Optional[str] = None  # When it will happen
    time_frame: Optional[TimeFrame] = None
    time_value: Optional[int] = None  # e.g., 3 for "3 months"
    
    # Context
    confidence: Confidence = Confidence.UNCERTAIN
    reasoning: Optional[str] = None  # Why they think this
    conditions: Optional[List[str]] = None  # "if X happens"
    
    # Metadata
    podcast_series: str = ""  # Podcast series name (e.g., "Bitcoin Dive Bar", "What Bitcoin Did")
    episode: str = ""  # Episode identifier
    episode_date: str = ""  # When episode aired
    timestamp: Optional[str] = None  # Time in episode (formatted)
    timestamp_start: Optional[float] = None  # Start time in seconds
    timestamp_end: Optional[float] = None  # End time in seconds
    youtube_link: Optional[str] = None  # Direct YouTube timestamp link
    raw_text: str = ""  # Original quote
    extraction_date: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Tracking
    prediction_id: Optional[str] = None  # Unique ID
    run_id: Optional[str] = None  # Run ID for tracking different extraction runs
    
    # Debug info
    timeframe_parsing_info: Optional[Dict] = None  # Track how timeframe was parsed
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds into YouTube timestamp format (1:23:45)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def generate_youtube_link(self, video_id: str) -> str:
        """Generate YouTube link with timestamp"""
        if self.timestamp_start is not None:
            timestamp_param = int(self.timestamp_start)
            return f"https://youtube.com/watch?v={video_id}&t={timestamp_param}s"
        return f"https://youtube.com/watch?v={video_id}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON storage"""
        data = asdict(self)
        # Convert enums to strings
        data['prediction_type'] = self.prediction_type.value
        data['confidence'] = self.confidence.value
        if self.time_frame:
            data['time_frame'] = self.time_frame.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Prediction':
        """Create from dictionary"""
        # Convert string enums back
        if 'prediction_type' in data:
            data['prediction_type'] = PredictionType(data['prediction_type'])
        if 'confidence' in data:
            data['confidence'] = Confidence(data['confidence'])
        if 'time_frame' in data and data['time_frame']:
            data['time_frame'] = TimeFrame(data['time_frame'])
        
        # Ensure value is a float
        if 'value' in data:
            data['value'] = float(data['value'])
            
        return cls(**data)

@dataclass
class Outcome:
    """Actual outcome for a prediction"""
    prediction_id: str
    actual_value: float
    actual_date: str
    was_correct: bool
    accuracy_score: Optional[float] = None  # How close were they
    notes: Optional[str] = None
    recorded_date: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class Speaker:
    """Track speaker accuracy over time"""
    name: str
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy_rate: float = 0.0
    average_confidence: float = 0.0
    favorite_assets: List[str] = field(default_factory=list)
    
    def update_stats(self, predictions: List[Prediction], outcomes: List[Outcome]):
        """Update speaker statistics"""
        speaker_preds = [p for p in predictions if p.speaker == self.name]
        self.total_predictions = len(speaker_preds)
        
        # Find outcomes for this speaker's predictions
        pred_ids = [p.prediction_id for p in speaker_preds]
        speaker_outcomes = [o for o in outcomes if o.prediction_id in pred_ids]
        
        self.correct_predictions = sum(1 for o in speaker_outcomes if o.was_correct)
        if self.total_predictions > 0:
            self.accuracy_rate = self.correct_predictions / self.total_predictions
        
        # Track favorite assets
        asset_counts = {}
        for pred in speaker_preds:
            asset_counts[pred.asset] = asset_counts.get(pred.asset, 0) + 1
        self.favorite_assets = sorted(asset_counts.keys(), 
                                    key=lambda x: asset_counts[x], 
                                    reverse=True)[:5]