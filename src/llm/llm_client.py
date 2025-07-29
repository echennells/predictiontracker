"""
LLM client for OpenAI API
"""
import os
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from openai import OpenAI

# Add config to path (go up from src/llm to root, then to config)
config_path = Path(__file__).parent.parent.parent / "config"
if config_path.exists():
    sys.path.insert(0, str(config_path))
else:
    # Fallback if structure is different
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    OPENAI_API_KEY, 
    OPENAI_API_BASE,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS
)


class OpenAIClient:
    """Client for interacting with OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = LLM_MODEL
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to OpenAI
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            response_format: Optional response format specification
            
        Returns:
            API response dict
        """
        try:
            # Handle o-model specific requirements (o3-mini, o4-mini)
            if "o3-mini" in self.model or "o4-mini" in self.model:
                # Convert system messages to developer role
                messages = [
                    {"role": "developer" if msg["role"] == "system" else msg["role"], "content": msg["content"]}
                    for msg in messages
                ]
                
                # o-models use max_completion_tokens instead of max_tokens
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "max_completion_tokens": max_tokens or 32768,  # o-models have higher limits
                }
                # No temperature or top_p for o-models
                
            else:
                # Standard models (GPT-4, GPT-4.1-nano, etc.)
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature or LLM_TEMPERATURE,
                    "max_tokens": max_tokens or LLM_MAX_TOKENS,
                }
            
            # Add response format if specified (for compatible models)
            if response_format and ("gpt-4o" in self.model or "gpt-3.5" in self.model or "gpt-4-turbo" in self.model):
                kwargs["response_format"] = response_format
            
            response = self.client.chat.completions.create(**kwargs)
            
            # Check if response was truncated
            if hasattr(response.choices[0], 'finish_reason'):
                if response.choices[0].finish_reason == 'length':
                    print(f"WARNING: Response truncated due to length limit for model {self.model}")
            
            # Convert to dict format similar to raw API response
            return {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }]
            }
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitError(f"Rate limited: {str(e)}")
            raise APIError(f"API request failed: {str(e)}")
    
    def extract_snippets(
        self,
        text: str,
        context: Optional[Dict[str, str]] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract snippets using a custom system prompt
        
        Args:
            text: Text to extract snippets from
            context: Optional context
            system_prompt: Custom system prompt for snippet extraction
            
        Returns:
            List of extracted snippets
        """
        if not system_prompt:
            system_prompt = "You are a helpful assistant that extracts information from text."
            
        user_prompt = f"""Find potential price predictions in this transcript section:

{text}

Context: {json.dumps(context) if context else 'No additional context'}
Extract generous snippets that might contain predictions. Output as JSON."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Use JSON mode for better reliability with GPT models (not all models support it)
        response_format = {"type": "json_object"} if ("gpt-4o" in self.model or "gpt-3.5" in self.model or "gpt-4-turbo" in self.model) else None
        response = self.chat_completion(
            messages=messages,
            response_format=response_format
        )
        
        try:
            content = response['choices'][0]['message']['content']
            
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            # Ensure we have a snippets array
            if isinstance(result, dict) and 'snippets' in result:
                return result['snippets']
            elif isinstance(result, list):
                return result
            else:
                return []
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing snippets response: {e}")
            print(f"Response length: {len(content)} characters")
            # Show more of the response to see where it cuts off
            print(f"Raw response start: {content[:1000]}...")
            if len(content) > 1000:
                print(f"Raw response end: ...{content[-500:]}")
            return []
    
    def extract_predictions(
        self,
        text: str,
        context: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract predictions from text using LLM
        
        Args:
            text: Text to extract predictions from
            context: Optional context (episode info, date, etc.)
            
        Returns:
            List of extracted predictions
        """
        system_prompt = """You are extracting price predictions from a Bitcoin podcast transcript.

TASK: Find predictions where a speaker states their own view about a future price.

DEFINITION OF PREDICTION:
- Speaker's own opinion (not "someone said" or "Jane Street thinks")  
- Specific price target (number)
- Future context (any timing, even implied)

ASSET MAPPING:
- "misty" → MSTY
- "similar", "silver", "sampler", "summer" → SMLR  
- "mst" → MSTR
- "big coin" → BTC
- "meta planet" → METAPLANET

PROCESS:
1. Scan for price mentions with future context
2. Identify the asset being discussed  
3. Confirm this is the speaker's own prediction
4. Extract details

OUTPUT: Use structured output with this exact schema:
{
  "predictions": [
    {
      "asset": "string (normalized ticker)",
      "price": number,
      "timeframe": "string (EXACTLY what speaker said - do not parse or interpret)",
      "confidence": "high|medium|low",
      "quote": "string (corrected slang)",
      "reasoning": "string (optional)",
      "timestamp": "string (HH:MM:SS or MM:SS)"
    }
  ]
}

TIMEFRAME EXTRACTION:
- Just copy EXACTLY what the speaker said about timing
- "tomorrow" → timeframe: "tomorrow"
- "next week" → timeframe: "next week"  
- "by August 15th" → timeframe: "by August 15th"
- "when we break 100k" → timeframe: "when we break 100k"
- "this bull run" → timeframe: "this bull run"
- If no timeframe mentioned → timeframe: ""

PRICE NORMALIZATION:
- BTC: "120" = 120000, "120k" = 120000
- BTC decimal notation: "1.4" = 140000, "1.7" = 170000, "1.25" = 125000
- Context clues: In Bitcoin discussions, small decimals (1.2, 1.7) usually mean hundreds of thousands
- Stocks: "200" = 200, "200k" = 200000
- For ranges, use midpoint: "130k-140k" = 135000

EXAMPLES:
✓ "I think SMLR goes to 50" → INCLUDE
✓ "Misty hits 100 next week" → INCLUDE  
✓ "We'll retrace to 80k" → INCLUDE (price drop prediction)
✗ "Jane Street said 120k" → SKIP (not speaker's view)
✗ "It's at 115k now" → SKIP (current price)

Be inclusive with timeframes - emotional context often implies timing."""

        user_prompt = f"""Extract predictions from this transcript:

{text}

Context: {json.dumps(context) if context else 'No additional context'}

Follow the PROCESS steps and output using the exact schema provided.
Be inclusive - casual mentions often have implied timeframes from context."""

        # Handle o-model role requirements
        if "o3-mini" in self.model or "o4-mini" in self.model:
            messages = [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        
        # Use JSON mode for better reliability (not for o-models or gpt-4.1-nano)
        response_format = {"type": "json_object"} if ("gpt-4o" in self.model or "gpt-3.5" in self.model or "gpt-4-turbo" in self.model) else None
        response = self.chat_completion(
            messages=messages,
            response_format=response_format
        )
        
        try:
            content = response['choices'][0]['message']['content']
            
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            # Ensure we have a predictions array
            if isinstance(result, dict) and 'predictions' in result:
                return result['predictions']
            elif isinstance(result, list):
                return result
            else:
                return []
                
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Failed to parse LLM response: {e}")
            return []
    
    def extract_predictions_raw(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Extract predictions using raw messages (for custom prompts)
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            List of extracted predictions
        """
        response = self.chat_completion(messages=messages)
        
        try:
            content = response['choices'][0]['message']['content']
            
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            # Ensure we have a predictions array
            if isinstance(result, dict) and 'predictions' in result:
                return result['predictions']
            elif isinstance(result, list):
                return result
            else:
                return []
                
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Failed to parse LLM response: {e}")
            return []


class ChunkProcessor:
    """Process large texts in chunks for LLM analysis"""
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        # Use config values if not specified
        from config import CHUNK_SIZE, CHUNK_OVERLAP
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.overlap = overlap or CHUNK_OVERLAP
    
    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk dicts with text and metadata
        """
        chunks = []
        text_length = len(text)
        
        start = 0
        chunk_num = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Try to find a sentence boundary near the end
            if end < text_length:
                # Look for sentence endings
                for delimiter in ['. ', '? ', '! ', '\n\n', '\n']:
                    last_delimiter = text.rfind(delimiter, start + self.chunk_size - 200, end)
                    if last_delimiter > start:
                        end = last_delimiter + len(delimiter)
                        break
            
            chunk_text = text[start:end]
            chunks.append({
                'text': chunk_text,
                'chunk_num': chunk_num,
                'start_char': start,
                'end_char': end,
                'is_first': chunk_num == 0,
                'is_last': end >= text_length
            })
            
            # Move start position, accounting for overlap
            start = end - self.overlap
            chunk_num += 1
        
        return chunks
    
    def merge_predictions(self, chunk_predictions: List[List[Dict]]) -> List[Dict]:
        """
        Merge predictions from multiple chunks, removing duplicates
        
        Args:
            chunk_predictions: List of prediction lists from each chunk
            
        Returns:
            Merged and deduplicated predictions
        """
        all_predictions = []
        seen_predictions = set()
        duplicate_count = 0
        
        for chunk_idx, predictions in enumerate(chunk_predictions):
            for pred in predictions:
                # Create a unique key for deduplication
                key = (
                    pred.get('asset', '').upper(),
                    float(pred.get('price', 0)),
                    pred.get('timeframe', '').lower()
                )
                
                if key not in seen_predictions and key[1] > 0:  # Valid price
                    seen_predictions.add(key)
                    all_predictions.append(pred)
                elif key[1] > 0:  # Valid price but duplicate
                    duplicate_count += 1
                    print(f"  Duplicate removed: {key[0]} ${key[1]:,.0f} (from chunk {chunk_idx + 1})")
        
        if duplicate_count > 0:
            print(f"  Total duplicates removed: {duplicate_count}")
        
        return all_predictions


class APIError(Exception):
    """Base exception for API errors"""
    pass


class RateLimitError(APIError):
    """Exception for rate limit errors"""
    pass