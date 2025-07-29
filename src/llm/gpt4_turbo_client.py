"""
GPT-4-turbo client - Most capable model for detailed extraction
Best for: Complex reasoning, nuanced predictions, high-stakes extraction
"""
from typing import List, Dict, Optional, Any
from openai import OpenAI
from datetime import datetime
import re
from .base_client import BaseLLMClient, ExtractionStage


class GPT4TurboClient(BaseLLMClient):
    """
    GPT-4-turbo: Most capable but expensive
    - $10.00/1M input tokens
    - $30.00/1M output tokens
    - 128k context window
    - Supports JSON mode
    - Best reasoning and nuance detection
    """
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)
    
    @property
    def model_name(self) -> str:
        return "gpt-4-turbo"
    
    @property
    def supports_json_mode(self) -> bool:
        return True
    
    @property
    def supports_temperature(self) -> bool:
        return True
    
    @property
    def context_window(self) -> int:
        return 128000  # 128k tokens
    
    @property
    def chunk_size(self) -> int:
        # Can handle large chunks but keep reasonable for cost
        # 40k chars ≈ 10k tokens
        return 40000
    
    @property
    def cost_per_million_input(self) -> float:
        return 10.00  # $10.00 per million input tokens
    
    @property
    def cost_per_million_output(self) -> float:
        return 30.00  # $30.00 per million output tokens
    
    def prepare_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Standard message format for GPT models"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call OpenAI API"""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.0,
            "max_tokens": max_tokens if max_tokens is not None else 4000,
        }
        
        # Add response format if specified
        if response_format:
            kwargs["response_format"] = response_format
        
        try:
            response = self.client.chat.completions.create(**kwargs)
            
            # Convert to expected format
            return {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }]
            }
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
    
    def deep_extract_predictions(self, text: str, context: Dict) -> List[Dict]:
        """
        Deep extraction with full context and nuanced understanding
        This is for high-value extraction where accuracy matters most
        """
        system_prompt = """You are an expert at extracting cryptocurrency price predictions from podcasts.

Your advanced capabilities:
1. Distinguish between serious predictions and jokes/sarcasm
2. Identify implied predictions from context
3. Understand conditional logic and scenarios
4. Detect confidence levels from tone and phrasing
5. Identify when speakers are quoting others vs making their own predictions
6. Parse complex timeframes and conditions

Extract ALL genuine predictions with full context and nuance.

For each prediction, provide:
- speaker: Who made it (track speakers across conversation)
- asset: The cryptocurrency or stock (normalize names)
- prediction_type: "price_target", "percentage", "direction", "range"
- target_value: The specific number mentioned
- timeframe: When they expect it (be specific)
- confidence: "very_high", "high", "medium", "low", "speculative"
- reasoning: Their stated reasoning
- conditions: Any "if" statements or prerequisites
- context_clues: How you determined this was a genuine prediction
- quote: The exact quote
- surrounding_context: Key context before/after

Output detailed JSON with all predictions found."""

        response = self.extract_for_stage(
            stage=ExtractionStage.DETAILED_EXTRACT,
            text=text,
            system_prompt=system_prompt,
            temperature=0.0,
            parse_json=True,
            max_tokens=8000  # Allow detailed output
        )
        
        predictions = []
        if response and 'predictions' in response:
            predictions = response['predictions']
        
        # Post-process with advanced logic
        return self._apply_advanced_processing(predictions, context)
    
    def _apply_advanced_processing(self, predictions: List[Dict], context: Dict) -> List[Dict]:
        """Apply advanced post-processing logic"""
        processed = []
        
        for pred in predictions:
            # Advanced asset recognition
            pred['asset'] = self._normalize_asset_advanced(pred.get('asset', ''))
            
            # Parse complex timeframes
            pred['parsed_timeframe'] = self._parse_timeframe(
                pred.get('timeframe', ''),
                context.get('episode_date', '')
            )
            
            # Calculate prediction score
            pred['quality_score'] = self._calculate_prediction_quality(pred)
            
            # Validate prediction logic
            if self._is_valid_prediction(pred):
                processed.append(pred)
        
        return processed
    
    def _normalize_asset_advanced(self, asset: str) -> str:
        """Advanced asset normalization with context understanding"""
        asset = asset.upper().strip()
        
        # Extended mapping
        mappings = {
            'BITCOIN': 'BTC', 'BIT COIN': 'BTC', 'BITCORN': 'BTC',
            'ETHEREUM': 'ETH', 'ETHER': 'ETH',
            'MICROSTRATEGY': 'MSTR', 'MICRO STRATEGY': 'MSTR', 'SAYLOR': 'MSTR',
            'COINBASE': 'COIN', 'COIN BASE': 'COIN',
            'MARATHON': 'MARA', 'MARATHON DIGITAL': 'MARA',
            'RIOT': 'RIOT', 'RIOT BLOCKCHAIN': 'RIOT',
            'SEMLER': 'SMLR', 'SEMLER SCIENTIFIC': 'SMLR',
            'SOLANA': 'SOL', 'CARDANO': 'ADA', 'CHAINLINK': 'LINK',
            'DOGECOIN': 'DOGE', 'DOGE COIN': 'DOGE',
        }
        
        for key, value in mappings.items():
            if key in asset:
                return value
        
        return asset
    
    def _parse_timeframe(self, timeframe: str, episode_date: str) -> Dict:
        """Parse complex timeframes into structured data"""
        timeframe = timeframe.lower()
        
        # Extract year if mentioned
        year_match = re.search(r'20\d{2}', timeframe)
        year = int(year_match.group()) if year_match else None
        
        # Parse relative timeframes
        if 'end of year' in timeframe or 'eoy' in timeframe:
            if year:
                return {'type': 'specific', 'date': f"{year}-12-31"}
            else:
                # Infer year from episode date
                ep_year = int(episode_date[:4]) if episode_date else datetime.now().year
                return {'type': 'specific', 'date': f"{ep_year}-12-31"}
        
        if 'next year' in timeframe:
            ep_year = int(episode_date[:4]) if episode_date else datetime.now().year
            return {'type': 'specific', 'date': f"{ep_year + 1}-12-31"}
        
        # Parse month references
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        for month_name, month_num in months.items():
            if month_name in timeframe:
                if year:
                    return {'type': 'specific', 'date': f"{year}-{month_num:02d}-01"}
                break
        
        # Parse relative durations
        if 'week' in timeframe:
            weeks = re.search(r'(\d+)\s*week', timeframe)
            if weeks:
                return {'type': 'relative', 'duration': f"{weeks.group(1)}_weeks"}
        
        if 'month' in timeframe:
            months = re.search(r'(\d+)\s*month', timeframe)
            if months:
                return {'type': 'relative', 'duration': f"{months.group(1)}_months"}
        
        return {'type': 'unknown', 'original': timeframe}
    
    def _calculate_prediction_quality(self, prediction: Dict) -> float:
        """Calculate quality score for prediction (0-1)"""
        score = 0.0
        
        # Has specific target
        if prediction.get('target_value'):
            score += 0.2
        
        # Has clear timeframe
        if prediction.get('parsed_timeframe', {}).get('type') != 'unknown':
            score += 0.2
        
        # Has reasoning
        if prediction.get('reasoning'):
            score += 0.2
        
        # High confidence
        confidence = prediction.get('confidence', 'low')
        if confidence in ['very_high', 'high']:
            score += 0.2
        elif confidence == 'medium':
            score += 0.1
        
        # Has supporting context
        if prediction.get('context_clues'):
            score += 0.1
        
        # Has exact quote
        if prediction.get('quote'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _is_valid_prediction(self, prediction: Dict) -> bool:
        """Validate if prediction meets quality threshold"""
        # Must have asset and some form of prediction
        if not prediction.get('asset') or not prediction.get('target_value'):
            return False
        
        # Quality score threshold
        if prediction.get('quality_score', 0) < 0.3:
            return False
        
        # Reject obvious errors (e.g., BTC to $1)
        if prediction.get('asset') == 'BTC':
            target = prediction.get('target_value', 0)
            if isinstance(target, (int, float)) and (target < 1000 or target > 10000000):
                return False
        
        return True
    
    def analyze_prediction_patterns(self, predictions: List[Dict]) -> Dict:
        """
        Analyze patterns across predictions
        Useful for understanding speaker tendencies and biases
        """
        analysis = {
            'speakers': {},
            'assets': {},
            'timeframes': {},
            'confidence_distribution': {},
            'average_targets': {}
        }
        
        for pred in predictions:
            speaker = pred.get('speaker', 'Unknown')
            asset = pred.get('asset', 'Unknown')
            confidence = pred.get('confidence', 'unknown')
            
            # Track by speaker
            if speaker not in analysis['speakers']:
                analysis['speakers'][speaker] = {
                    'count': 0,
                    'assets': {},
                    'avg_confidence': []
                }
            
            analysis['speakers'][speaker]['count'] += 1
            analysis['speakers'][speaker]['assets'][asset] = \
                analysis['speakers'][speaker]['assets'].get(asset, 0) + 1
            
            # Track by asset
            if asset not in analysis['assets']:
                analysis['assets'][asset] = {
                    'count': 0,
                    'targets': [],
                    'speakers': set()
                }
            
            analysis['assets'][asset]['count'] += 1
            if isinstance(pred.get('target_value'), (int, float)):
                analysis['assets'][asset]['targets'].append(pred['target_value'])
            analysis['assets'][asset]['speakers'].add(speaker)
        
        # Calculate averages
        for asset, data in analysis['assets'].items():
            if data['targets']:
                analysis['average_targets'][asset] = sum(data['targets']) / len(data['targets'])
            data['speakers'] = list(data['speakers'])  # Convert set to list for JSON
        
        return analysis