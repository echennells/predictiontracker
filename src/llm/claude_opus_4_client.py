"""
Claude Opus 4 client via OpenRouter - Anthropic's most powerful model
Best for: Complex extraction requiring state-of-the-art capabilities
"""
from typing import List, Dict, Optional, Any
from openai import OpenAI
from .base_client import BaseLLMClient, ExtractionStage


class ClaudeOpus4Client(BaseLLMClient):
    """
    Claude Opus 4: Anthropic's flagship model (via OpenRouter)
    - $15/1M input tokens
    - $75/1M output tokens
    - 200k context window
    - World's best coding model (72.5% on SWE-bench)
    - Supports extended agent workflows
    """
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        # Use OpenRouter base URL
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    
    @property
    def model_name(self) -> str:
        return "anthropic/claude-opus-4"
    
    @property
    def supports_json_mode(self) -> bool:
        # Claude doesn't support OpenAI's JSON mode
        return False
    
    @property
    def supports_temperature(self) -> bool:
        return True
    
    @property
    def context_window(self) -> int:
        return 200000  # 200k tokens
    
    @property
    def chunk_size(self) -> int:
        # Can handle very large chunks due to 200k context
        # 150k chars ≈ 40k tokens, leaving plenty of room
        return 150000
    
    @property
    def cost_per_million_input(self) -> float:
        return 15.0  # $15 per million input tokens
    
    @property
    def cost_per_million_output(self) -> float:
        return 75.0  # $75 per million output tokens
    
    def prepare_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Standard message format for Claude via OpenRouter"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def extract_for_stage(
        self,
        stage: ExtractionStage,
        text: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        parse_json: bool = True
    ) -> Any:
        """
        Extract predictions for a specific stage
        
        Claude Opus 4 excels at:
        - Complex reasoning and analysis
        - Understanding nuanced context
        - Accurate timeframe parsing
        - High-quality extraction with fewer false positives
        """
        if not system_prompt:
            system_prompt = self._get_default_prompt_for_stage(stage)
        
        messages = self.prepare_messages(system_prompt, text)
        
        try:
            # Track usage
            input_tokens = self.count_tokens(system_prompt + text)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            output_tokens = self.count_tokens(content)
            
            # Update token tracking
            self.update_token_usage({
                'input': input_tokens,
                'output': output_tokens,
                'calls': 1
            })
            
            if parse_json:
                return self.parse_json_response(content)
            return content
            
        except Exception as e:
            print(f"[ERROR] Claude Opus 4 extraction failed: {e}")
            raise
    
    def _get_default_prompt_for_stage(self, stage: ExtractionStage) -> str:
        """Get stage-specific prompts optimized for Claude Opus 4"""
        
        if stage == ExtractionStage.SNIPPET_SCAN:
            return """You are analyzing a podcast transcript to find potential price predictions.

Your task: Identify ALL sections where speakers discuss future prices of assets.

Be VERY inclusive - include any mention of:
- Specific future prices (even casual mentions)
- Price targets or ranges
- Conditional price scenarios
- Timeframe-based predictions

For each potential prediction area, extract a generous snippet (include full context).

Output as JSON:
{
  "snippets": [
    {
      "text": "the snippet text with full context",
      "confidence": "high|medium|low",
      "timestamp": "HH:MM:SS"
    }
  ]
}"""

        elif stage == ExtractionStage.PREDICTION_EXTRACT:
            return """Extract cryptocurrency price predictions from this text.

Focus on speaker's OWN predictions (not third-party mentions).

Requirements:
- Asset being predicted
- Specific price target
- Timeframe mentioned (copy EXACTLY as stated)
- Speaker's confidence level

Output as JSON:
{
  "predictions": [
    {
      "asset": "ticker symbol",
      "price": number,
      "timeframe": "EXACT quote from speaker",
      "confidence": "high|medium|low",
      "quote": "relevant quote",
      "timestamp": "HH:MM:SS"
    }
  ]
}"""

        else:  # DETAILED_EXTRACT
            return """You are performing deep analysis of cryptocurrency predictions with a focus on parsing timeframes.

Your PRIMARY task is to interpret and parse timeframe strings into structured data.

For each prediction, analyze the timeframe and determine:
1. The type: days, weeks, months, years, specific_date, end_of_year, conditional, unclear
2. The numeric value (if applicable)
3. The predicted date (calculate based on episode date if relative)

Parse these patterns:
- "next week" → type: "weeks", value: 1
- "in 3 months" → type: "months", value: 3
- "by August 15th" → type: "specific_date", predicted_date: "2024-08-15"
- "end of year" → type: "end_of_year"
- "when X happens" → type: "conditional"

Use your advanced reasoning to handle complex or ambiguous timeframes.

Output refined predictions with parsed timeframe data."""