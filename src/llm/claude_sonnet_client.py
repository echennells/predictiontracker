"""
Claude 3 Sonnet client via OpenRouter - Anthropic's balanced model
Best for: Different perspective, excellent reasoning, nuanced understanding
"""
from typing import List, Dict, Optional, Any
from openai import OpenAI
import json
from .base_client import BaseLLMClient, ExtractionStage


class ClaudeSonnetClient(BaseLLMClient):
    """
    Claude 3 Sonnet: Anthropic's balanced model
    - $3.00/1M input tokens
    - $15.00/1M output tokens
    - 200k context window
    - No native JSON mode (but very good at following instructions)
    - Excellent at reasoning and nuance
    """
    
    def __init__(self, api_key: str, use_openrouter: bool = True):
        super().__init__(api_key)
        if use_openrouter:
            # Use OpenRouter
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "config"))
            import config
            
            self.client = OpenAI(
                api_key=api_key or config.OPENROUTER_API_KEY,
                base_url=config.OPENROUTER_API_BASE
            )
            self.use_openrouter = True
        else:
            # Direct Anthropic API (would need anthropic package)
            raise NotImplementedError("Direct Anthropic API not implemented, use OpenRouter")
    
    @property
    def model_name(self) -> str:
        # OpenRouter uses different model names
        if self.use_openrouter:
            return "anthropic/claude-3-sonnet"
        return "claude-3-sonnet-20240229"
    
    @property
    def supports_json_mode(self) -> bool:
        return False  # No native JSON mode, but reliable with instructions
    
    @property
    def supports_temperature(self) -> bool:
        return True
    
    @property
    def context_window(self) -> int:
        return 200000  # 200k tokens
    
    @property
    def chunk_size(self) -> int:
        # Large chunks for good context
        # 150k chars ≈ 37k tokens
        return 150000
    
    @property
    def cost_per_million_input(self) -> float:
        return 3.00  # $3.00 per million input tokens
    
    @property
    def cost_per_million_output(self) -> float:
        return 15.00  # $15.00 per million output tokens
    
    def prepare_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Claude uses a different message format"""
        # Claude combines system into the first user message
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        return [{"role": "user", "content": combined_prompt}]
    
    def call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call API via OpenRouter"""
        try:
            # OpenRouter uses OpenAI-compatible API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # OpenRouter handles system messages properly
                temperature=temperature if temperature is not None else 0.0,
                max_tokens=max_tokens if max_tokens is not None else 4000,
            )
            
            # Convert to expected format
            return {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }]
            }
        except Exception as e:
            raise Exception(f"Claude API call failed: {str(e)}")
    
    def extract_with_reasoning(self, text: str, context: Dict) -> List[Dict]:
        """
        Claude excels at step-by-step reasoning
        Use this for complex extraction requiring nuanced understanding
        """
        system_prompt = """You are extracting cryptocurrency price predictions from podcast transcripts.

Claude's approach:
1. First, identify all speakers and their roles
2. Then, scan for prediction-like statements
3. For each potential prediction, reason through:
   - Is this their own prediction or quoting someone else?
   - Is this serious or joking?
   - What's the confidence level based on their language?
   - Are there implicit conditions?

Think step-by-step and show your reasoning.

Output format:
First, your reasoning process.
Then, a JSON block with the extracted predictions:

```json
{
  "predictions": [
    {
      "speaker": "name",
      "asset": "BTC",
      "prediction": {"value": 150000, "timeframe": "2025"},
      "confidence": "high",
      "reasoning": "They said 'I'm very confident' and gave specific reasons"
    }
  ]
}
```"""

        response = self.extract_for_stage(
            stage=ExtractionStage.FOCUSED_SCAN,
            text=text,
            system_prompt=system_prompt,
            temperature=0.0,
            parse_json=False  # We'll parse manually due to reasoning text
        )
        
        # Extract JSON from response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
            try:
                result = json.loads(json_str)
                return result.get('predictions', [])
            except json.JSONDecodeError:
                print("Failed to parse Claude's JSON output")
                return []
        
        return []
    
    def validate_predictions(self, predictions: List[Dict], transcript_context: str) -> List[Dict]:
        """
        Use Claude's reasoning to validate predictions
        Good for catching false positives
        """
        system_prompt = """You are validating cryptocurrency predictions extracted from a podcast.

For each prediction, determine if it's genuine by checking:
1. Is the quote accurate to the transcript?
2. Is it actually a prediction (not just discussing current prices)?
3. Is the speaker making their own prediction (not quoting others)?
4. Is the context interpreted correctly?

Be strict - only validate genuine predictions.

Output format:
```json
{
  "validated": [
    {
      ...original prediction...,
      "validation": {
        "is_valid": true,
        "confidence": 0.95,
        "notes": "Clear prediction with specific target"
      }
    }
  ],
  "rejected": [
    {
      ...original prediction...,
      "reason": "Why this was rejected"
    }
  ]
}
```"""

        user_prompt = f"""Transcript context:
{transcript_context}

Predictions to validate:
{json.dumps(predictions, indent=2)}

Validate each prediction against the transcript context."""

        response = self.extract_for_stage(
            stage=ExtractionStage.FOCUSED_SCAN,
            text=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            parse_json=False
        )
        
        # Extract JSON from response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
            try:
                result = json.loads(json_str)
                return result.get('validated', [])
            except json.JSONDecodeError:
                return predictions  # Return original if parsing fails
        
        return predictions
    
    def extract_conditional_predictions(self, text: str) -> List[Dict]:
        """
        Claude is particularly good at understanding conditional logic
        Use for complex if/then predictions
        """
        system_prompt = """Focus on extracting conditional predictions from this podcast transcript.

Look for patterns like:
- "If X happens, then Bitcoin will..."
- "Assuming Y, we could see..."
- "In the scenario where..."
- "Unless Z occurs..."

For each conditional prediction, extract:
1. The condition(s)
2. The prediction if condition is met
3. Any alternative outcomes mentioned
4. The speaker's assessment of condition likelihood

Output format:
```json
{
  "conditional_predictions": [
    {
      "speaker": "name",
      "condition": "if ETF is approved",
      "then_prediction": {"asset": "BTC", "target": 150000},
      "else_prediction": {"asset": "BTC", "target": 80000},
      "condition_likelihood": "high",
      "quote": "exact quote"
    }
  ]
}
```"""

        response = self.extract_for_stage(
            stage=ExtractionStage.DETAILED_EXTRACT,
            text=text,
            system_prompt=system_prompt,
            temperature=0.0,
            parse_json=False
        )
        
        # Extract JSON from response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
            try:
                result = json.loads(json_str)
                return result.get('conditional_predictions', [])
            except json.JSONDecodeError:
                return []
        
        return []