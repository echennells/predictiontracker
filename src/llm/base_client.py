"""
Base class for all LLM clients with stage-aware functionality
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
import json
import tiktoken
from enum import Enum


class ExtractionStage(Enum):
    """Different stages of extraction with different cost/accuracy tradeoffs"""
    INITIAL_SCAN = "initial_scan"       # Cheap, fast, find potential areas
    FOCUSED_SCAN = "focused_scan"       # Medium cost, extract specific snippets
    DETAILED_EXTRACT = "detailed_extract"  # Expensive, high accuracy extraction


class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.token_usage = {
            ExtractionStage.INITIAL_SCAN: {'input': 0, 'output': 0, 'calls': 0},
            ExtractionStage.FOCUSED_SCAN: {'input': 0, 'output': 0, 'calls': 0},
            ExtractionStage.DETAILED_EXTRACT: {'input': 0, 'output': 0, 'calls': 0}
        }
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier"""
        pass
    
    @property
    @abstractmethod
    def supports_json_mode(self) -> bool:
        """Whether this model supports JSON response format"""
        pass
    
    @property
    @abstractmethod
    def supports_temperature(self) -> bool:
        """Whether this model supports temperature parameter"""
        pass
    
    @property
    @abstractmethod
    def context_window(self) -> int:
        """Maximum context window in tokens"""
        pass
    
    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Recommended chunk size in characters for this model"""
        pass
    
    @property
    @abstractmethod
    def cost_per_million_input(self) -> float:
        """Cost per million input tokens in USD"""
        pass
    
    @property
    @abstractmethod
    def cost_per_million_output(self) -> float:
        """Cost per million output tokens in USD"""
        pass
    
    @abstractmethod
    def prepare_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """
        Prepare messages in the format required by this model
        Some models use 'system' role, others use 'developer'
        """
        pass
    
    @abstractmethod
    def call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make the actual API call"""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens for this model"""
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fallback to cl100k_base encoding for newer models
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    def track_usage(self, stage: ExtractionStage, input_tokens: int, output_tokens: int):
        """Track token usage for cost calculation"""
        self.token_usage[stage]['input'] += input_tokens
        self.token_usage[stage]['output'] += output_tokens
        self.token_usage[stage]['calls'] += 1
    
    def get_stage_cost(self, stage: ExtractionStage) -> float:
        """Calculate cost for a specific stage"""
        usage = self.token_usage[stage]
        input_cost = (usage['input'] / 1_000_000) * self.cost_per_million_input
        output_cost = (usage['output'] / 1_000_000) * self.cost_per_million_output
        return input_cost + output_cost
    
    def get_total_cost(self) -> float:
        """Calculate total cost across all stages"""
        return sum(self.get_stage_cost(stage) for stage in ExtractionStage)
    
    def extract_for_stage(
        self,
        stage: ExtractionStage,
        text: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        parse_json: bool = True
    ) -> Any:
        """
        Main extraction method that handles different stages
        
        Args:
            stage: The extraction stage (affects prompting and parsing)
            text: Input text to process
            system_prompt: System prompt for this stage
            temperature: Temperature override
            max_tokens: Max tokens override
            parse_json: Whether to parse response as JSON
            
        Returns:
            Parsed response (dict/list if JSON, str otherwise)
        """
        # Prepare user prompt based on stage
        if stage == ExtractionStage.INITIAL_SCAN:
            user_prompt = f"Scan this text for relevant content:\n\n{text}"
        elif stage == ExtractionStage.FOCUSED_SCAN:
            user_prompt = f"Extract specific information from this text:\n\n{text}"
        else:  # DETAILED_EXTRACT
            user_prompt = f"Perform detailed extraction on this text:\n\n{text}"
        
        # Count input tokens
        messages = self.prepare_messages(system_prompt, user_prompt)
        input_text = system_prompt + user_prompt
        input_tokens = self.count_tokens(input_text)
        
        # Determine response format
        response_format = None
        if parse_json and self.supports_json_mode:
            response_format = {"type": "json_object"}
        
        # Make API call
        response = self.call_api(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
        
        # Extract content
        content = response['choices'][0]['message']['content']
        
        # Count output tokens
        output_tokens = self.count_tokens(content)
        
        # Track usage
        self.track_usage(stage, input_tokens, output_tokens)
        
        # Parse response if needed
        if parse_json:
            return self._parse_json_response(content)
        return content
    
    def _parse_json_response(self, content: str) -> Any:
        """Parse JSON response, handling common edge cases"""
        # Remove markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Content: {content[:500]}...")
            return None
    
    def chunk_text(self, text: str, overlap: int = 1000) -> List[Tuple[str, int, int]]:
        """
        Chunk text based on model's recommended chunk size
        
        Returns list of (chunk_text, start_pos, end_pos) tuples
        """
        chunks = []
        chunk_size = self.chunk_size
        
        if len(text) <= chunk_size:
            return [(text, 0, len(text))]
        
        pos = 0
        while pos < len(text):
            end = min(pos + chunk_size, len(text))
            chunk = text[pos:end]
            chunks.append((chunk, pos, end))
            
            if end >= len(text):
                break
                
            # Move position with overlap
            pos = end - overlap
        
        return chunks