"""
GPT-4-0125-preview client - Latest GPT-4 with massive context window
Best for: Processing entire transcripts at once, maintaining full context
"""
from typing import List, Dict, Optional, Any
from openai import OpenAI
from .base_client import BaseLLMClient, ExtractionStage


class GPT41Client(BaseLLMClient):
    """
    GPT-4-0125-preview: Latest GPT-4 with 1M token context
    - $2.00/1M input tokens (cheaper than turbo!)
    - $6.00/1M output tokens
    - 1M token context window (can process entire episodes)
    - Supports JSON mode
    - Updated knowledge and capabilities
    """
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)
    
    @property
    def model_name(self) -> str:
        return "gpt-4-0125-preview"
    
    @property
    def supports_json_mode(self) -> bool:
        return True
    
    @property
    def supports_temperature(self) -> bool:
        return True
    
    @property
    def context_window(self) -> int:
        return 1000000  # 1M tokens!
    
    @property
    def chunk_size(self) -> int:
        # Can handle massive chunks - entire transcripts
        # 800k chars ≈ 200k tokens, leaving plenty of room
        return 800000
    
    @property
    def cost_per_million_input(self) -> float:
        return 2.00  # $2.00 per million input tokens
    
    @property
    def cost_per_million_output(self) -> float:
        return 6.00  # $6.00 per million output tokens
    
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
    
    def process_entire_transcript(self, transcript: str, episode_info: Dict) -> List[Dict]:
        """
        Process entire transcript in one go - the main advantage of 1M context
        No chunking needed, maintains full conversation context
        """
        system_prompt = """You are extracting cryptocurrency price predictions from a complete podcast transcript.

With the full context available, you can:
1. Track speakers throughout the entire conversation
2. Understand callbacks and references to earlier points
3. Identify running jokes vs serious predictions
4. See the full arc of arguments and reasoning
5. Catch predictions that span multiple segments

Extract ALL genuine price predictions with their full context.

For each prediction:
{
  "speaker": "Name of person making prediction",
  "asset": "BTC/ETH/etc",
  "prediction": {
    "type": "price_target|percentage|range",
    "value": 150000,
    "timeframe": "end of 2025",
    "confidence": "high|medium|low"
  },
  "context": {
    "timestamp": "approximate time in transcript",
    "reasoning": "their stated logic",
    "conditions": "any if/then conditions",
    "references_earlier": true/false,
    "quote": "exact quote"
  }
}

Be thorough but only include genuine predictions by the speakers themselves."""

        # Single API call for entire transcript
        response = self.extract_for_stage(
            stage=ExtractionStage.DETAILED_EXTRACT,
            text=transcript,
            system_prompt=system_prompt,
            temperature=0.0,
            parse_json=True,
            max_tokens=8000
        )
        
        predictions = []
        if response and 'predictions' in response:
            for pred in response['predictions']:
                # Add episode metadata
                pred['episode'] = episode_info.get('title', '')
                pred['episode_date'] = episode_info.get('date', '')
                predictions.append(pred)
        
        return predictions
    
    def cross_reference_predictions(self, transcript: str, initial_predictions: List[Dict]) -> List[Dict]:
        """
        Use full context to verify and enhance predictions found by other models
        This is where the 1M context really shines
        """
        system_prompt = """You have the full podcast transcript and a list of predictions found by another model.

Your task:
1. Verify each prediction exists in the transcript
2. Add missing context using the full conversation
3. Identify any missed predictions
4. Correct any misattributions or errors
5. Note if speakers later retract or modify their predictions

For each prediction, enhance or correct it based on the full context.

Output the verified and enhanced predictions list."""

        user_prompt = f"""Full transcript:
{transcript}

Initial predictions to verify:
{json.dumps(initial_predictions, indent=2)}

Verify and enhance these predictions using the full context."""

        response = self.extract_for_stage(
            stage=ExtractionStage.FOCUSED_SCAN,
            text=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            parse_json=True,
            max_tokens=8000
        )
        
        if response and 'predictions' in response:
            return response['predictions']
        
        return initial_predictions  # Fallback to original if processing fails
    
    def extract_speaker_profiles(self, transcript: str) -> Dict[str, Dict]:
        """
        Build comprehensive speaker profiles from full transcript
        Useful for attribution and understanding prediction context
        """
        system_prompt = """Analyze the full transcript to build speaker profiles.

For each distinct speaker, identify:
1. Their name/identifier
2. Their role (host, guest, etc)
3. Their general stance on crypto (bullish/bearish/neutral)
4. Their expertise areas
5. Speaking patterns that help identify them
6. Any stated biases or positions

Output:
{
  "speakers": {
    "Speaker Name": {
      "role": "host|guest",
      "stance": "bullish|bearish|neutral",
      "expertise": ["bitcoin", "trading", "macroeconomics"],
      "identifying_phrases": ["I think", "basically"],
      "prediction_style": "specific|vague|conditional"
    }
  }
}"""

        response = self.extract_for_stage(
            stage=ExtractionStage.INITIAL_SCAN,
            text=transcript,
            system_prompt=system_prompt,
            temperature=0.0,
            parse_json=True
        )
        
        if response and 'speakers' in response:
            return response['speakers']
        
        return {}