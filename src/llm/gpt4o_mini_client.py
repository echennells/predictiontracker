"""
GPT-4o-mini client - Optimized for cheap, fast initial scanning
Best for: Finding potential prediction areas in large transcripts
"""
from typing import List, Dict, Optional, Any
from openai import OpenAI
from .base_client import BaseLLMClient, ExtractionStage


class GPT4OMiniClient(BaseLLMClient):
    """
    GPT-4o-mini: Cheapest option with good capabilities
    - $0.15/1M input tokens
    - $0.60/1M output tokens
    - 128k context window
    - Supports JSON mode
    - Fast response times
    """
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)
    
    @property
    def model_name(self) -> str:
        return "gpt-4o-mini"
    
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
        # Use large chunks for initial scanning
        # 100k chars ≈ 25k tokens, leaving plenty of room
        return 100000
    
    @property
    def cost_per_million_input(self) -> float:
        return 0.15  # $0.15 per million input tokens
    
    @property
    def cost_per_million_output(self) -> float:
        return 0.60  # $0.60 per million output tokens
    
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
            "max_tokens": max_tokens if max_tokens is not None else 2000,
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
    
    def scan_for_predictions(self, text: str, context: Dict[str, Any]) -> List[int]:
        """
        Optimized method for initial scanning - returns line numbers only
        Much cheaper than extracting full snippets
        """
        system_prompt = """You are scanning a podcast transcript for cryptocurrency price predictions.
Your ONLY job is to identify line numbers where speakers make predictions about future prices.

Look for:
- Specific price targets (e.g., "Bitcoin to 100k")
- Percentage predictions (e.g., "up 50% by December")
- Time-based predictions (e.g., "by end of year")
- Conditional predictions (e.g., "if X happens, then Y price")

Output ONLY line numbers, one per line. Nothing else."""

        # For initial scan, we can be more aggressive with chunk size
        chunks = self.chunk_text(text, overlap=500)  # Less overlap for scanning
        all_line_numbers = []
        
        for chunk, start_pos, end_pos in chunks:
            # Add line numbers to chunk
            lines = chunk.split('\n')
            numbered_chunk = '\n'.join(f"{i}: {line}" for i, line in enumerate(lines))
            
            response = self.extract_for_stage(
                stage=ExtractionStage.INITIAL_SCAN,
                text=numbered_chunk,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=500,  # Just numbers, so small output
                parse_json=False  # Simple text output
            )
            
            # Parse line numbers
            for line in response.split('\n'):
                line = line.strip()
                if line.isdigit():
                    # Adjust for chunk offset
                    actual_line = int(line) + (text[:start_pos].count('\n'))
                    all_line_numbers.append(actual_line)
        
        return sorted(set(all_line_numbers))
    
    def extract_snippets(self, text: str, line_numbers: List[int], context_lines: int = 10) -> List[Dict]:
        """
        Extract snippets around identified line numbers
        More focused than full text scanning
        """
        system_prompt = """Extract generous snippets around cryptocurrency price predictions.
Include enough context to understand:
- Who is speaking
- What asset they're discussing
- The prediction details
- Any conditions or timeframes

Output JSON:
{
  "snippets": [
    {
      "text": "The actual snippet text with context",
      "line_start": 100,
      "line_end": 120,
      "confidence": "high/medium/low",
      "reason": "Brief description of the prediction"
    }
  ]
}"""

        lines = text.split('\n')
        snippets = []
        
        # Group nearby line numbers to avoid overlap
        grouped_lines = []
        current_group = []
        
        for num in sorted(line_numbers):
            if not current_group or num - current_group[-1] <= context_lines * 2:
                current_group.append(num)
            else:
                grouped_lines.append(current_group)
                current_group = [num]
        
        if current_group:
            grouped_lines.append(current_group)
        
        # Extract snippet for each group
        for group in grouped_lines:
            start = max(0, min(group) - context_lines)
            end = min(len(lines), max(group) + context_lines)
            
            snippet_lines = lines[start:end]
            snippet_text = '\n'.join(f"{i}: {line}" for i, line in enumerate(snippet_lines, start))
            
            response = self.extract_for_stage(
                stage=ExtractionStage.FOCUSED_SCAN,
                text=snippet_text,
                system_prompt=system_prompt,
                temperature=0.0,
                parse_json=True
            )
            
            if response and 'snippets' in response:
                snippets.extend(response['snippets'])
        
        return snippets