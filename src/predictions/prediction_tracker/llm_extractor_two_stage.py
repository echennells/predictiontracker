"""
Two-stage LLM prediction extractor:
1. GPT-4o-mini finds potential prediction snippets (cheap, fast)
2. GPT-4o-mini extracts actual predictions from snippets (with full context)
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import gc
import time
import tiktoken
import os
from pathlib import Path


class QuotaExceededException(Exception):
    """Raised when OpenAI API quota is exceeded"""
    pass


# Import new modular LLM clients
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.llm.gpt4o_mini_client import GPT4OMiniClient
from src.llm.gpt4o_client import GPT4OClient
from src.llm.gpt4_turbo_client import GPT4TurboClient
from src.llm.gpt4_1_client import GPT41Client
from src.llm.claude_sonnet_client import ClaudeSonnetClient
from src.llm.claude_opus_4_client import ClaudeOpus4Client
from src.llm.base_client import ExtractionStage
from .models import Prediction, PredictionType, Confidence, TimeFrame
from .sanity_checker import PredictionSanityChecker


class TwoStageLLMExtractor:
    """Three-stage prediction extractor with token usage tracking and dry run support"""
    
    def __init__(self):
        # Load config to get stage-specific models
        import config
        
        # Create clients based on configured models
        self.stage1_model = config.STAGE1_MODEL
        self.stage2_model = config.STAGE2_MODEL
        self.stage3_model = config.STAGE3_MODEL
        self.stage3_dry_run = config.STAGE3_DRY_RUN
        
        # Initialize clients based on model names
        self.snippet_client = self._create_client(self.stage1_model)
        self.prediction_client = self._create_client(self.stage2_model)
        self.refinement_client = self._create_client(self.stage3_model) if not self.stage3_dry_run else None
        
        print(f"[STAGE CONFIG] Stage 1: {self.stage1_model}, Stage 2: {self.stage2_model}, Stage 3: {self.stage3_model} (dry_run={self.stage3_dry_run})")
        
        # Token tracking
        self.token_usage = {
            'snippet_extraction': {'input': 0, 'output': 0, 'calls': 0},
            'prediction_extraction': {'input': 0, 'output': 0, 'calls': 0},
            'refinement': {'input': 0, 'output': 0, 'calls': 0}
        }
        
        # Dry run log for Stage 3
        self.stage3_dry_run_log = []
        
        # Debug logging
        self.debug_log_dir = Path("logs/debug")
        self.debug_log_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log_file = None
        self.current_episode = None
        
        # Asset normalization mapping (updated with discovered slang)
        self.asset_map = {
            # Crypto
            'bitcoin': 'BTC', 'btc': 'BTC', 'big coin': 'BTC', 'bigcoin': 'BTC',
            'ethereum': 'ETH', 'eth': 'ETH', 'ether': 'ETH',
            'solana': 'SOL', 'sol': 'SOL',
            'chainlink': 'LINK', 'link': 'LINK',
            'cardano': 'ADA', 'ada': 'ADA',
            'ripple': 'XRP', 'xrp': 'XRP',
            'doge': 'DOGE', 'dogecoin': 'DOGE',
            'matic': 'MATIC', 'polygon': 'MATIC',
            
            # MSTR and related
            'mstr': 'MSTR', 'microstrategy': 'MSTR', 'micro strategy': 'MSTR', 'mst': 'MSTR',
            'msty': 'MSTY', 'misty': 'MSTY',  # The leveraged ETF
            
            # SMLR variants (from our scan)
            'smlr': 'SMLR', 'semler': 'SMLR', 'similar': 'SMLR', 'sampler': 'SMLR', 'summer': 'SMLR',
            'silver': 'SMLR',  # Common mistranscription when they say "similar"
            
            # Other Bitcoin companies
            'coin': 'COIN', 'coinbase': 'COIN',
            'mara': 'MARA', 'marathon': 'MARA',
            'riot': 'RIOT',
            'clsk': 'CLSK', 'cleanspark': 'CLSK',
            'hut': 'HUT', 'hut8': 'HUT',
            'bitf': 'BITF', 'bitfarms': 'BITF',
            'hive': 'HIVE',
            'gbtc': 'GBTC', 'grayscale': 'GBTC',
            'meta planet': 'METAPLANET', 'metaplanet': 'METAPLANET',
            
            # People (not assets, but good to recognize)
            'sailor': 'SAYLOR', 'saylor': 'SAYLOR',
        }
    
    def _create_client(self, model_name: str):
        """Create appropriate client based on model name"""
        import config
        
        if model_name == "gpt-4o-mini":
            return GPT4OMiniClient(config.OPENAI_API_KEY)
        elif model_name == "gpt-4o":
            return GPT4OClient(config.OPENAI_API_KEY)
        elif model_name == "gpt-4-turbo":
            return GPT4TurboClient(config.OPENAI_API_KEY)
        elif model_name == "gpt-4-0125-preview":
            return GPT41Client(config.OPENAI_API_KEY)
        elif model_name == "anthropic/claude-opus-4":
            return ClaudeOpus4Client(config.OPENROUTER_API_KEY)
        elif model_name.startswith("anthropic/") or model_name.startswith("claude"):
            return ClaudeSonnetClient(config.OPENROUTER_API_KEY, use_openrouter=True)
        else:
            # Default to GPT-4o-mini for unknown models
            print(f"[WARNING] Unknown model '{model_name}', defaulting to gpt-4o-mini")
            return GPT4OMiniClient(config.OPENAI_API_KEY)
    
    def _init_debug_log(self, episode_title: str):
        """Initialize debug log for this episode"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_title = "".join(c for c in episode_title if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
        self.debug_log_file = self.debug_log_dir / f"debug_{timestamp}_{safe_title}.json"
        self.current_episode = episode_title
        self.debug_data = {
            "episode": episode_title,
            "timestamp": timestamp,
            "models": {
                "snippet": self.stage1_model,
                "prediction": self.stage2_model,
                "refinement": self.stage3_model,
                "stage3_dry_run": self.stage3_dry_run
            },
            "snippets": [],
            "predictions": [],
            "errors": []
        }
    
    def _log_debug(self, stage: str, data: Dict):
        """Log debug data to file"""
        if self.debug_log_file and self.debug_data:
            if stage == "snippet":
                self.debug_data["snippets"].append(data)
            elif stage == "prediction":
                self.debug_data["predictions"].append(data)
            elif stage == "error":
                self.debug_data["errors"].append(data)
            
            # Write to file after each update
            with open(self.debug_log_file, 'w') as f:
                json.dump(self.debug_data, f, indent=2)
    
    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """Count tokens for a given text and model"""
        try:
            # Use cl100k_base encoding for GPT-4 models
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except:
            # Fallback to approximate count
            return len(text) // 4
    
    def find_prediction_locations(self, text: str, context: Dict) -> List[Dict]:
        """
        Stage 1a: Find LOCATIONS where predictions might be
        Returns simple positions, no JSON needed
        """
        system_prompt = """Find locations in this transcript where speakers mention ANY PRICE with future context.

CRITICAL: When in doubt, INCLUDE the position.
Better to have false positives than miss real predictions.

Cast a wide net - look for ANY price mention with future context:
- "Bitcoin 120k" ✓ (include even without explicit timing)
- "SMLR to 50" ✓ (include casual predictions) 
- "I think we see 200" ✓ (include vague future references)
- "misty hits 100" ✓ (slang is fine)
- "similar going to 50" ✓ (mistranscriptions too)

Your job: Find ALL potential predictions, even borderline cases.
Stage 2 will do the precise filtering.

INCLUDE ANY mention of:
- Price numbers with assets (even without timing)
- "going to X", "hits X", "to X", "sees X"
- Future-sounding price mentions
- Slang terms: "misty" (MSTY), "similar/silver" (SMLR), "mst" (MSTR), "big coin" (BTC)

ONLY SKIP:
- Current prices ("it's at 115k now")
- Past prices ("was at 60k")

Return ONLY character positions as numbers, one per line.
Example output:
12450
45200
67890

Just the numbers, nothing else!"""

        all_positions = []
        
        # Process in 50k sections to keep responses small
        section_size = 50000
        overlap = 2500
        
        for start in range(0, len(text), section_size - overlap):
            end = min(start + section_size, len(text))
            section = text[start:end]
            
            if len(section.strip()) < 100:
                continue
                
            print(f"\n[LOCATION FINDING] Processing section {start:,}-{end:,} chars...")
            
            # Track token usage
            user_content = f"Find prediction locations in this transcript section:\n\n{section}"
            input_tokens = self.count_tokens(system_prompt + user_content, self.stage1_model)
            self.token_usage['snippet_extraction']['input'] += input_tokens
            self.token_usage['snippet_extraction']['calls'] += 1
            
            try:
                # Use new client API
                response = self.snippet_client.extract_for_stage(
                    stage=ExtractionStage.INITIAL_SCAN,
                    text=section,
                    system_prompt=system_prompt,
                    temperature=0.0,
                    parse_json=False
                )
                
                content = response.strip() if isinstance(response, str) else str(response)
                
                # Track output tokens
                output_tokens = self.count_tokens(content, self.stage1_model)
                self.token_usage['snippet_extraction']['output'] += output_tokens
                
                # Parse simple number list
                for line in content.split('\n'):
                    line = line.strip()
                    if line and line.isdigit():
                        pos = int(line) + start  # Adjust for section offset
                        all_positions.append(pos)
                
                lines = content.split('\n')
                num_predictions = len([l for l in lines if l.strip().isdigit()])
                print(f"   Found {num_predictions} predictions in this section")
                
            except Exception as e:
                error_str = str(e)
                print(f"   Error in section {start}-{end}: {e}")
                
                # Check if this is a quota error
                if 'insufficient_quota' in error_str or '429' in error_str:
                    raise QuotaExceededException(
                        f"OpenAI API quota exceeded. Please add credits to your account or wait for quota reset.\n"
                        f"Original error: {error_str}"
                    )
                continue
        
        # Deduplicate positions within 3000 chars (to avoid ANY overlapping snippets)
        # Since we extract 3000 chars around each position (1500 before, 1500 after),
        # positions need to be at least 3000 chars apart to have zero overlap
        unique_positions = []
        for pos in sorted(all_positions):
            if not unique_positions or pos - unique_positions[-1] >= 3000:
                unique_positions.append(pos)
        
        print(f"\n[LOCATION FINDING] Total: {len(unique_positions)} unique prediction locations")
        print(f"  (Deduplicated from {len(all_positions)} raw positions)")
        
        # Convert to expected format
        return [{"position": pos} for pos in unique_positions]
    
    def test_api_access(self) -> bool:
        """Test if API access is working before processing episodes"""
        try:
            print("\n[API TEST] Testing OpenAI API access...")
            test_response = self.snippet_client.extract_for_stage(
                stage=ExtractionStage.INITIAL_SCAN,
                text="Test API access",
                system_prompt="Reply with 'OK'",
                temperature=0.0,
                parse_json=False,
                max_tokens=10
            )
            print("[API TEST] ✓ API access confirmed")
            return True
        except Exception as e:
            error_str = str(e)
            if 'insufficient_quota' in error_str or '429' in error_str:
                print("\n[API ERROR] ❌ OpenAI API quota exceeded!")
                print("Please add credits to your account at: https://platform.openai.com/account/billing")
                print(f"Error details: {error_str}")
                raise QuotaExceededException("Cannot proceed - API quota exceeded")
            else:
                print(f"[API ERROR] ❌ API test failed: {error_str}")
                raise
    
    def extract_snippet_around_position(self, text: str, position: int, context_chars: int = 3000) -> Dict:
        """
        Extract ~3 minutes of context around a position (about 500-800 words = ~3000 chars)
        """
        # We want context_chars total, with the prediction roughly in the middle
        # So extract context_chars/2 before and after the position
        half_context = context_chars // 2
        
        # Calculate start and end with bounds checking
        start = max(0, position - half_context)
        end = min(len(text), position + half_context)
        
        # If we hit the beginning, extend the end to maintain total context
        if start == 0:
            end = min(len(text), context_chars)
        # If we hit the end, extend the start
        elif end == len(text):
            start = max(0, len(text) - context_chars)
        
        # Find sentence boundaries to avoid cutting mid-sentence
        if start > 0:
            # Look for sentence start
            for i in range(start, min(start + 200, position)):
                if i > 0 and text[i-1:i+1] in ['. ', '? ', '! ', '\n\n']:
                    start = i + 1
                    break
        
        if end < len(text):
            # Look for sentence end
            for i in range(max(position, end - 200), end):
                if i < len(text) - 1 and text[i:i+2] in ['. ', '? ', '! ', '\n\n']:
                    end = i + 2
                    break
        
        return {
            'text': text[start:end],
            'char_start': start,
            'char_end': end
        }
    
    def extract_snippets(self, text: str, context: Dict, start_offset: int = 0) -> List[Dict]:
        """
        Stage 1: Use cheap model to find potential prediction snippets
        
        Returns list of snippets with their positions in the text
        """
        # Already using gpt-4o-mini from __init__, no need to override
        
        system_prompt = """You are a snippet extractor for a Bitcoin/crypto podcast. 
Your job is to find parts of the conversation that MIGHT contain price predictions made BY THE SPEAKERS THEMSELVES.

IMPORTANT: Skip sections where they're quoting others ("Jane Street said", "Saylor thinks", "the guy said", "someone mentioned").

Look for sections where THE PODCAST SPEAKERS discuss:
- Future prices (going to, will hit, target, headed to, could reach, might see)
- Price levels (120k, 150k, $50, "two hundred", "triple digits", etc)
- Timeframes (next week, by end of year, in 3 months, this cycle, soon)
- Technical analysis (support, resistance, breakout, retracement, consolidation)
- Bullish/bearish sentiment with numbers or levels
- Asset mentions (BTC, Bitcoin, MSTR, "Misty", ETH, COIN, SMLR, Ethereum, MicroStrategy, etc.)
- Sequential predictions ("first this, then that", "after X then Y", "before we see")
- Conditional language ("if we break", "when volume picks up", "once we hit", "assuming")
- Market movements ("pump to", "dump to", "retrace to", "bounce off")
- Comparisons ("like last time when", "similar to 2021", "remember when it hit")

Extract snippets where SPEAKERS THEMSELVES (not quoting others) discuss:
- Specific price numbers (120k, $50, 200k - NOT vague like "100 plus")
- WITH future timeframes (next week, by end of year, this cycle, etc)
- For specific assets (BTC, Bitcoin, MSTR, SMLR, etc)

SKIP snippets that:
- Quote external sources ("Jane Street said", "according to X")
- Have NO timeframe context
- Use vague prices ("going higher", "100 plus", "triple digits")

IMPORTANT: Extract moderate context around predictions - approximately 30-45 SECONDS of conversation (roughly 150-200 words). 
This ensures the second model has enough context to understand:
- What asset is being discussed (could be mentioned earlier)
- The full context of the prediction
- Any conditions or reasoning

Include AT LEAST 3-5 sentences before and after the prediction mention to capture context.

Output SIMPLE JSON:
{
  "snippets": [
    {
      "text": "Moderate snippet: Include 30-45 seconds of conversation (150-200 words) with context before and after the prediction",
      "char_start": 0,
      "char_end": 1000,
      "reason": "mentions BTC going to 130k next week"
    }
  ]
}

CRITICAL: Keep total JSON response under 8000 characters to avoid truncation!

BE EXTREMELY GENEROUS - it's much better to extract too many snippets than miss predictions.
When in doubt, ALWAYS include it! False positives are fine, false negatives are absolutely not acceptable.
Even vague discussions about future prices should be included."""

        # Process in chunks if needed
        # GPT-4o-mini has 128k token context, so we use reasonable chunks
        chunk_size = 100000  # ~25k tokens per chunk, leaving room for prompts
        snippets = []
        
        # Count input tokens
        system_tokens = self.count_tokens(system_prompt, self.snippet_client.model)
        
        # Remove the old chunking logic - we'll always use the full text
        # The extract_snippets_in_sections method handles large texts
        if False:  # Disabled old chunking
            print(f"\n[SNIPPET EXTRACTION] Processing {len(text):,} chars in chunks...")
            
            for i in range(0, len(text), chunk_size - 10000):  # 10k overlap
                chunk = text[i:i + chunk_size]
                
                user_prompt = f"""Find potential price predictions in this transcript section:

{chunk}

Context: {json.dumps(context)}

CRITICAL: Extract VERY GENEROUS snippets - approximately 3 MINUTES of conversation (500-800 words) around each potential prediction.
Include AT LEAST 10-15 sentences before and after any price mention to ensure full context about what asset is being discussed."""
                
                # Track tokens
                user_tokens = self.count_tokens(user_prompt, self.snippet_client.model)
                self.token_usage['snippet_extraction']['input'] += system_tokens + user_tokens
                self.token_usage['snippet_extraction']['calls'] += 1
                
                # Make API call with custom system prompt
                chunk_snippets = self.snippet_client.extract_snippets(chunk, context, system_prompt)
                
                # Track output tokens (approximate)
                output_tokens = self.count_tokens(str(chunk_snippets), self.snippet_client.model)
                self.token_usage['snippet_extraction']['output'] += output_tokens
                
                snippets.extend(chunk_snippets)
                
                print(f"   Chunk {i//chunk_size + 1}: Found {len(chunk_snippets)} potential snippets")
                time.sleep(0.5)  # Rate limiting
        else:
            user_prompt = f"""Find potential price predictions in this transcript:

{text}

Context: {json.dumps(context)}

CRITICAL: Extract VERY GENEROUS snippets - approximately 3 MINUTES of conversation (500-800 words) around each potential prediction.
Include AT LEAST 10-15 sentences before and after any price mention to ensure full context about what asset is being discussed."""
            
            # Track tokens
            user_tokens = self.count_tokens(user_prompt, self.stage1_model)
            self.token_usage['snippet_extraction']['input'] += system_tokens + user_tokens
            self.token_usage['snippet_extraction']['calls'] += 1
            
            # Use new client API for snippet extraction
            try:
                response = self.snippet_client.extract_for_stage(
                    stage=ExtractionStage.INITIAL_SCAN,
                    text=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.0,
                    parse_json=True
                )
                snippets = response.get('snippets', []) if isinstance(response, dict) else []
            except Exception as e:
                print(f"[ERROR] Snippet extraction failed: {e}")
                snippets = []
            
            # Adjust character positions if we have a start_offset
            if start_offset > 0:
                for snippet in snippets:
                    if 'char_start' in snippet:
                        snippet['char_start'] += start_offset
                    if 'char_end' in snippet:
                        snippet['char_end'] += start_offset
            
            # Track output tokens
            output_tokens = self.count_tokens(str(snippets), self.snippet_client.model)
            self.token_usage['snippet_extraction']['output'] += output_tokens
        
        print(f"\n[SNIPPET EXTRACTION] Found {len(snippets)} potential prediction snippets")
        print(f"[TOKEN USAGE] Snippet extraction - Input: {self.token_usage['snippet_extraction']['input']:,}, Output: {self.token_usage['snippet_extraction']['output']:,}")
        
        # Log debug data for all snippets
        for i, snippet in enumerate(snippets):
            self._log_debug("snippet", {
                "index": i,
                "text": snippet.get('text', ''),
                "reason": snippet.get('reason', ''),
                "char_start": snippet.get('char_start', 0),
                "char_end": snippet.get('char_end', 0),
                "tokens": self.count_tokens(snippet.get('text', ''), self.stage1_model)
            })
        
        return snippets
    
    def extract_predictions_from_snippets(self, snippets: List[Dict], episode_info: Dict) -> List[Prediction]:
        """
        Stage 2: Use expensive model to extract actual predictions from snippets
        """
        all_raw_predictions = []
        
        print(f"\n[PREDICTION EXTRACTION] Processing {len(snippets)} snippets with expensive model...")
        
        # Filter out empty snippets
        valid_snippets = [s for s in snippets if s.get('text', '')]
        
        if not valid_snippets:
            print("No valid snippets to process")
            return []
        
        # Check if we can use batch processing
        if hasattr(self.prediction_client, 'extract_predictions'):
            print(f"\n[STAGE 2] Using batch prediction extraction for {len(valid_snippets)} snippets")
            
            # Count tokens for all snippets
            total_input_tokens = 0
            for snippet in valid_snippets:
                system_tokens = 500  # Approximate system prompt size
                user_tokens = self.count_tokens(snippet.get('text', ''), self.stage2_model)
                total_input_tokens += system_tokens + user_tokens
            
            self.token_usage['prediction_extraction']['input'] += total_input_tokens
            self.token_usage['prediction_extraction']['calls'] += len(valid_snippets)
            
            # Extract predictions for all snippets at once (enables concurrent processing)
            try:
                predictions = self.prediction_client.extract_predictions(valid_snippets, episode_info)
                
                # Count output tokens
                output_tokens = self.count_tokens(str(predictions), self.stage2_model)
                self.token_usage['prediction_extraction']['output'] += output_tokens
                
                if predictions:
                    print(f"\n[STAGE 2] ✓ Found {len(predictions)} total predictions across all snippets")
                    
                    # Log debug data for each prediction
                    for i, pred in enumerate(predictions):
                        self._log_debug("raw_prediction", {
                            "prediction_index": i,
                            "snippet_text": pred.get('quote', ''),
                            "predictions_found": 1,
                            "raw_predictions": [pred],
                            "tokens_used": {
                                "input": 0,  # Already counted above
                                "output": 0
                            }
                        })
                        
                        # Add snippet timing info to each prediction
                        # This will need to be matched later since we're not processing one snippet at a time
                        pred['snippet_timestamp_start'] = None
                        pred['snippet_timestamp_end'] = None
                    
                    all_raw_predictions.extend(predictions)
                else:
                    print(f"\n[STAGE 2] ✗ No predictions found in any snippet")
                    
            except Exception as e:
                print(f"[ERROR] Batch prediction extraction failed: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to sequential processing
                print("[FALLBACK] Falling back to sequential processing...")
                
                # Process snippets one by one (original logic)
                for i, snippet in enumerate(valid_snippets):
                    snippet_text = snippet.get('text', '')
                    
                    print(f"\n   Snippet {i+1}/{len(valid_snippets)} ({len(snippet_text)} chars)")
                    print(f"   Reason: {snippet.get('reason', 'No reason given')}")
                    
                    # Count tokens for this snippet  
                    system_tokens = 500  # Approximate system prompt size
                    user_tokens = self.count_tokens(snippet_text, self.stage2_model)
                    self.token_usage['prediction_extraction']['input'] += system_tokens + user_tokens
                    self.token_usage['prediction_extraction']['calls'] += 1
                    
                    # Extract predictions using Stage 2 model
                    context = {
                        'episode_title': episode_info.get('title', ''),
                        'episode_date': episode_info.get('date', ''),
                        'snippet_position': snippet.get('position', 'unknown'),
                        'snippet_timestamp_start': snippet.get('timestamp_start'),
                        'snippet_timestamp_end': snippet.get('timestamp_end')
                    }
                    
                    try:
                        predictions = self.prediction_client.extract_predictions([snippet], context)
                    except Exception as e:
                        print(f"[ERROR] Prediction extraction failed for snippet {i+1}: {e}")
                        predictions = []
                    
                    # Track output tokens
                    output_tokens = self.count_tokens(str(predictions), self.stage2_model)
                    self.token_usage['prediction_extraction']['output'] += output_tokens
                    
                    print(f"   Found {len(predictions)} predictions in this snippet")
                    
                    # Log debug data for this snippet's predictions
                    self._log_debug("prediction", {
                        "snippet_index": i,
                        "snippet_text": snippet_text,
                        "snippet_reason": snippet.get('reason', ''),
                        "predictions_found": len(predictions),
                        "raw_predictions": predictions,
                        "tokens_used": {
                            "input": user_tokens,
                            "output": output_tokens
                        }
                    })
                    
                    # Add snippet timing info to each prediction
                    for pred in predictions:
                        pred['snippet_timestamp_start'] = snippet.get('timestamp_start')
                        pred['snippet_timestamp_end'] = snippet.get('timestamp_end')
                    
                    # Just collect all predictions without deduplication
                    all_raw_predictions.extend(predictions)
                    
                    # Rate limiting
                    time.sleep(0.5)
        else:
            # Use sequential processing for clients that don't support batch extraction
            print(f"\n[STAGE 2] Using sequential processing (client doesn't support batch extraction)")
            
            for i, snippet in enumerate(valid_snippets):
                snippet_text = snippet.get('text', '')
                
                print(f"\n   Snippet {i+1}/{len(valid_snippets)} ({len(snippet_text)} chars)")
                print(f"   Reason: {snippet.get('reason', 'No reason given')}")
                
                # Count tokens for this snippet  
                system_tokens = 500  # Approximate system prompt size
                user_tokens = self.count_tokens(snippet_text, self.stage2_model)
                self.token_usage['prediction_extraction']['input'] += system_tokens + user_tokens
                self.token_usage['prediction_extraction']['calls'] += 1
                
                # Extract predictions using Stage 2 model
                try:
                    system_prompt = self._get_stage2_system_prompt()
                    response = self.prediction_client.extract_for_stage(
                        stage=ExtractionStage.FOCUSED_SCAN,
                        text=snippet_text,
                        system_prompt=system_prompt,
                        temperature=0.0,
                        parse_json=True
                    )
                    predictions = response.get('predictions', []) if isinstance(response, dict) else []
                except Exception as e:
                    print(f"[ERROR] Prediction extraction failed: {e}")
                    predictions = []
                
                # Track output tokens
                output_tokens = self.count_tokens(str(predictions), self.stage2_model)
                self.token_usage['prediction_extraction']['output'] += output_tokens
                
                print(f"   Found {len(predictions)} predictions in this snippet")
                
                # Add snippet timing info to each prediction
                for pred in predictions:
                    pred['snippet_timestamp_start'] = snippet.get('timestamp_start')
                    pred['snippet_timestamp_end'] = snippet.get('timestamp_end')
                
                # Just collect all predictions without deduplication
                all_raw_predictions.extend(predictions)
                
                # Rate limiting
                time.sleep(0.5)
        
        print(f"\n[PREDICTION EXTRACTION] Total raw predictions: {len(all_raw_predictions)}")
        
        # Now deduplicate and convert to prediction objects
        final_predictions = self._deduplicate_and_convert_predictions(all_raw_predictions, episode_info)
        
        print(f"[PREDICTION EXTRACTION] After deduplication: {len(final_predictions)} unique predictions")
        print(f"[TOKEN USAGE] Prediction extraction - Input: {self.token_usage['prediction_extraction']['input']:,}, Output: {self.token_usage['prediction_extraction']['output']:,}")
        
        # Stage 3: Deep refinement (or dry run)
        if final_predictions and self.stage3_model:
            final_predictions = self._stage3_refinement(final_predictions, episode_info)
        
        return final_predictions
    
    def _deduplicate_and_convert_predictions(self, raw_predictions: List[Dict], episode_info: Dict) -> List[Prediction]:
        """Deduplicate raw predictions and convert to Prediction objects"""
        seen_predictions = set()
        final_predictions = []
        
        print(f"\n[DEDUPLICATION] Processing {len(raw_predictions)} raw predictions...")
        
        for pred in raw_predictions:
            # Create a comprehensive key based on the raw prediction data
            # Include asset, price, and a normalized quote to catch duplicates
            quote = pred.get('quote', '').strip()[:100]  # First 100 chars of quote
            asset = pred.get('asset', '').upper()
            price = pred.get('price', 0)
            
            # Comprehensive key - asset, price, and quote
            # This prevents missing different price predictions with similar wording
            key = f"{asset}|{price}|{quote}"
            
            if key not in seen_predictions:
                seen_predictions.add(key)
                
                # Try to create prediction object
                prediction_obj = self._create_prediction_object(pred, episode_info)
                if prediction_obj:
                    # Validate that we have a quote - essential for verification and timestamp matching
                    if not prediction_obj.raw_text or len(prediction_obj.raw_text.strip()) < 10:
                        print(f"   ✗ Rejected: {prediction_obj.asset} ${prediction_obj.value:,.0f} - No quote provided")
                        self._log_debug("error", {
                            "stage": "prediction_validation", 
                            "reason": "No quote provided by LLM",
                            "asset": prediction_obj.asset,
                            "value": prediction_obj.value,
                            "prediction": pred
                        })
                        continue
                    
                    final_predictions.append(prediction_obj)
                    try:
                        # Format time_frame safely (it's an enum)
                        time_frame_str = prediction_obj.time_frame.value if prediction_obj.time_frame else "No timeframe"
                        print(f"   ✓ {prediction_obj.asset} ${prediction_obj.value:,.0f} - {time_frame_str}")
                    except Exception as e:
                        print(f"   ✓ {prediction_obj.asset} ${prediction_obj.value} - {prediction_obj.time_frame} (formatting error: {e})")
                else:
                    print(f"   ✗ Failed to convert: {asset} {pred.get('price', 'unknown price')}")
                    self._log_debug("error", {
                        "stage": "prediction_conversion",
                        "raw_prediction": pred,
                        "reason": "Failed to create prediction object"
                    })
            else:
                print(f"   - Duplicate skipped: {asset} (same quote)")
        
        return final_predictions
    
    def extract_predictions_from_file(self, transcript_file: str, episode_info: Dict) -> List[Prediction]:
        """
        Main entry point: Extract predictions using two-stage approach
        """
        print(f"\n=== TWO-STAGE PREDICTION EXTRACTION ===")
        print(f"Episode: {episode_info.get('title', 'Unknown')}")
        
        # Initialize debug logging for this episode
        self._init_debug_log(episode_info.get('title', 'Unknown'))
        
        # Reset token usage for this episode
        self.token_usage = {
            'snippet_extraction': {'input': 0, 'output': 0, 'calls': 0},
            'prediction_extraction': {'input': 0, 'output': 0, 'calls': 0},
            'refinement': {'input': 0, 'output': 0, 'calls': 0}
        }
        
        # Reset dry run log
        self.stage3_dry_run_log = []
        
        # Load transcript text
        with open(transcript_file, 'r') as f:
            data = json.load(f)
            text = data.get('text', '')
            segments = data.get('segments', [])
        
        # Store segments for quote matching later
        self.current_segments = segments
        
        print(f"Transcript length: {len(text):,} characters")
        
        # Stage 1: Find prediction locations, then extract 3-minute snippets
        print("\n[STAGE 1] Finding prediction locations...")
        locations = self.find_prediction_locations(text, episode_info)
        
        if not locations:
            print("No potential predictions found")
            return []
        
        # Extract 3 minutes of context around each location
        print(f"\n[STAGE 1] Extracting 3-minute context around {len(locations)} locations...")
        
        # Build character position to segment mapping
        char_to_segment = {}
        if segments:
            char_pos = 0
            for seg_idx, segment in enumerate(segments):
                seg_text = segment.get('text', '')
                seg_start = segment.get('start', 0)
                seg_end = segment.get('end', 0)
                seg_len = len(seg_text)
                for i in range(seg_len):
                    char_to_segment[char_pos + i] = {
                        'segment_idx': seg_idx,
                        'start_time': seg_start,
                        'end_time': seg_end
                    }
                char_pos += seg_len + 1  # +1 for space between segments
        
        snippets = []
        for i, loc in enumerate(locations):
            position = loc.get('position', 0)
            snippet = self.extract_snippet_around_position(text, position)
            
            # Find the timestamp for this snippet based on character position
            snippet_center = (snippet['char_start'] + snippet['char_end']) // 2
            if snippet_center in char_to_segment:
                seg_info = char_to_segment[snippet_center]
                snippet['timestamp_start'] = seg_info['start_time']
                snippet['timestamp_end'] = seg_info['end_time']
                snippet['segment_idx'] = seg_info['segment_idx']
            else:
                # Fallback: find nearest character position
                nearest_pos = min(char_to_segment.keys(), key=lambda x: abs(x - snippet_center)) if char_to_segment else None
                if nearest_pos and abs(nearest_pos - snippet_center) < 500:  # Within 500 chars
                    seg_info = char_to_segment[nearest_pos]
                    snippet['timestamp_start'] = seg_info['start_time']
                    snippet['timestamp_end'] = seg_info['end_time']
                    snippet['segment_idx'] = seg_info['segment_idx']
            
            # Extract a preview from the snippet text for debugging
            preview_start = max(0, position - snippet['char_start'])
            preview_end = min(preview_start + 100, len(snippet['text']))
            snippet['reason'] = snippet['text'][preview_start:preview_end].strip()
            snippets.append(snippet)
            print(f"   Location {i+1}: position {position:,} → snippet {snippet['char_start']:,}-{snippet['char_end']:,} ({len(snippet['text'])} chars)")
        
        if not snippets:
            print("No potential prediction snippets found")
            return []
        
        # Stage 2: Extract predictions from snippets with expensive model
        predictions = self.extract_predictions_from_snippets(snippets, episode_info)
        
        # Timestamp matching is now handled during prediction creation using snippet timestamps
        print(f"\n[TIMESTAMP INFO] All predictions have timestamps from their source snippets")
        
        # Print final token usage summary
        print(f"\n=== TOKEN USAGE SUMMARY ===")
        print(f"Snippet Extraction ({self.stage1_model}):")
        print(f"  Calls: {self.token_usage['snippet_extraction']['calls']}")
        print(f"  Input tokens: {self.token_usage['snippet_extraction']['input']:,}")
        print(f"  Output tokens: {self.token_usage['snippet_extraction']['output']:,}")
        print(f"  Total tokens: {self.token_usage['snippet_extraction']['input'] + self.token_usage['snippet_extraction']['output']:,}")
        
        print(f"\nPrediction Extraction ({self.stage2_model}):")
        print(f"  Calls: {self.token_usage['prediction_extraction']['calls']}")
        print(f"  Input tokens: {self.token_usage['prediction_extraction']['input']:,}")
        print(f"  Output tokens: {self.token_usage['prediction_extraction']['output']:,}")
        print(f"  Total tokens: {self.token_usage['prediction_extraction']['input'] + self.token_usage['prediction_extraction']['output']:,}")
        
        if self.token_usage['refinement']['calls'] > 0 or self.stage3_dry_run:
            print(f"\nRefinement ({self.stage3_model}) {'[DRY RUN]' if self.stage3_dry_run else ''}:")
            print(f"  Calls: {self.token_usage['refinement']['calls']}")
            print(f"  Input tokens: {self.token_usage['refinement']['input']:,}")
            print(f"  Output tokens: {self.token_usage['refinement']['output']:,}")
            print(f"  Total tokens: {self.token_usage['refinement']['input'] + self.token_usage['refinement']['output']:,}")
        
        # Cost estimation using actual model costs
        stage1_cost = 0
        stage2_cost = 0
        stage3_cost = 0
        
        # Calculate Stage 1 cost
        if hasattr(self.snippet_client, 'cost_per_million_input'):
            stage1_cost = (self.token_usage['snippet_extraction']['input'] / 1_000_000) * self.snippet_client.cost_per_million_input + \
                         (self.token_usage['snippet_extraction']['output'] / 1_000_000) * self.snippet_client.cost_per_million_output
        
        # Calculate Stage 2 cost
        if hasattr(self.prediction_client, 'cost_per_million_input'):
            stage2_cost = (self.token_usage['prediction_extraction']['input'] / 1_000_000) * self.prediction_client.cost_per_million_input + \
                         (self.token_usage['prediction_extraction']['output'] / 1_000_000) * self.prediction_client.cost_per_million_output
        
        # Calculate Stage 3 cost (if not dry run)
        if not self.stage3_dry_run and self.refinement_client and hasattr(self.refinement_client, 'cost_per_million_input'):
            stage3_cost = (self.token_usage['refinement']['input'] / 1_000_000) * self.refinement_client.cost_per_million_input + \
                         (self.token_usage['refinement']['output'] / 1_000_000) * self.refinement_client.cost_per_million_output
        
        print(f"\nEstimated costs:")
        print(f"  Stage 1 ({self.stage1_model}): ${stage1_cost:.4f}")
        print(f"  Stage 2 ({self.stage2_model}): ${stage2_cost:.4f}")
        if self.stage3_dry_run:
            print(f"  Stage 3 ({self.stage3_model}): $0.0000 [DRY RUN]")
        else:
            print(f"  Stage 3 ({self.stage3_model}): ${stage3_cost:.4f}")
        print(f"  Total: ${stage1_cost + stage2_cost + stage3_cost:.4f}")
        
        # Final debug log summary
        if self.debug_data:
            self.debug_data["summary"] = {
                "total_snippets": len(self.debug_data["snippets"]),
                "total_predictions": len(predictions),
                "token_usage": self.token_usage,
                "cost": {
                    "snippet_extraction": stage1_cost,
                    "prediction_extraction": stage2_cost,
                    "refinement": stage3_cost,
                    "total": stage1_cost + stage2_cost + stage3_cost
                },
                "stage3_dry_run": self.stage3_dry_run,
                "stage3_dry_run_log": self.stage3_dry_run_log if self.stage3_dry_run else []
            }
            self._log_debug("summary", {})
            print(f"\n[DEBUG LOG] Saved to: {self.debug_log_file}")
        
        # Sanity check predictions before returning
        if predictions:
            sanity_checker = PredictionSanityChecker(debug_logger=self._log_debug if self.debug_data else None)
            predictions = sanity_checker.check_predictions(predictions, episode_info)
        
        return predictions
    
    def _find_quote_in_segments(self, quote: str, segments: List[Dict]) -> Optional[Dict]:
        """Find which segment(s) contain a quote and return timing info using key phrase matching"""
        import re
        
        quote_lower = quote.lower().strip()
        
        if not quote_lower or len(quote_lower) < 10:
            return None
        
        # Extract key phrases for matching
        # 1. Price patterns (most distinctive)
        price_patterns = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:k|m|b|dollars?|bucks?|grand))?', quote_lower)
        
        # 2. Asset mentions
        asset_patterns = re.findall(r'\b(?:btc|bitcoin|eth|ethereum|smlr|similar|mstr|microstrategy|ibit|i\s*bet|msty|misty)\b', quote_lower)
        
        # 3. Action words (less common, more distinctive)
        action_patterns = re.findall(r'\b(?:goes?\s+to|hit(?:s|ting)?|reach(?:es|ing)?|pump(?:s|ing)?|moon(?:s|ing)?|crash(?:es|ing)?|dump(?:s|ing)?)\b', quote_lower)
        
        # 4. Extract 3-5 word phrases as anchors (skip very common words)
        common_words = {'the', 'a', 'an', 'is', 'it', 'to', 'and', 'or', 'but', 'in', 'on', 'at', 'for', 'with', 'this', 'that', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she', 'we', 'they', 'if', 'then', 'so'}
        words = quote_lower.split()
        key_phrases = []
        
        # Find 3-4 word sequences with at least one non-common word
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if any(word not in common_words and len(word) > 2 for word in words[i:i+3]):
                key_phrases.append(phrase)
                
        # Prioritize phrases with prices or distinctive words
        key_phrases.sort(key=lambda p: sum(1 for pattern in price_patterns + asset_patterns if pattern in p), reverse=True)
        
        best_match = None
        best_score = 0
        
        # Search segments
        for i, segment in enumerate(segments):
            segment_text = segment.get('text', '').lower().strip()
            
            if not segment_text:
                continue
                
            score = 0
            matches = []
            
            # Check for price matches (highest weight)
            for price in price_patterns[:3]:  # Check top 3 price patterns
                if price in segment_text:
                    score += 3
                    matches.append(f"price:{price}")
                    
            # Check for asset matches (high weight)
            for asset in asset_patterns[:2]:  # Check top 2 assets
                if asset in segment_text:
                    score += 2
                    matches.append(f"asset:{asset}")
                    
            # Check for key phrases (medium weight)
            for phrase in key_phrases[:5]:  # Check top 5 phrases
                if phrase in segment_text:
                    score += 1
                    matches.append(f"phrase:{phrase}")
                    
            # Bonus for segments that contain multiple matches close together
            if len(matches) >= 2:
                score += 2
                
            # Normalize score by segment length to prefer shorter segments
            if score > 0:
                score = score * (100 / (len(segment_text) + 100))
                
            if score > best_score:
                best_score = score
                best_match = {
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'segment_index': i,
                    'confidence': score,
                    'matches': matches
                }
                
            # Check if quote spans multiple segments
            if score > 0 and i < len(segments) - 1:
                next_segment = segments[i + 1]
                next_text = next_segment.get('text', '').lower().strip()
                
                # Check if key parts of quote continue in next segment
                combined_text = segment_text + " " + next_text
                combined_score = score
                
                for phrase in key_phrases:
                    if phrase in combined_text and phrase not in segment_text:
                        combined_score += 1
                        
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = {
                        'start': segment.get('start', 0),
                        'end': next_segment.get('end', 0),
                        'segment_index': i,
                        'confidence': combined_score,
                        'matches': matches + ['spans_segments']
                    }
        
        # Debug output
        if best_match and hasattr(self, 'debug_data'):
            print(f"   [QUOTE MATCH] Found quote with confidence {best_match['confidence']:.2f}")
            print(f"   [QUOTE MATCH] Matches: {best_match['matches']}")
            print(f"   [QUOTE MATCH] Segment {best_match['segment_index']}: {best_match['start']:.1f}s - {best_match['end']:.1f}s")
        
        return best_match if best_score > 0.5 else None
    
    def _create_prediction_object(self, pred_data: Dict, episode_info: Dict) -> Optional[Prediction]:
        """Convert LLM output to Prediction object"""
        try:
            asset_raw = pred_data.get('asset', '').lower()
            asset = self.asset_map.get(asset_raw, asset_raw.upper())
            
            if not asset or len(asset) > 10:
                print(f"   [DEBUG] Asset '{asset}' rejected (empty or too long)")
                return None
            
            # LLM should return clean numeric values, but handle cases where it doesn't
            raw_price = pred_data.get('price')
            if raw_price is None:
                print(f"   [DEBUG] Price is None/null from LLM, skipping prediction")
                return None
            
            try:
                price = float(raw_price)
                print(f"   [DEBUG] Price from LLM: {price}")
            except (ValueError, TypeError) as e:
                print(f"   [DEBUG] Price parsing error for '{raw_price}': {e}")
                return None
            
            if price <= 0:
                print(f"   [DEBUG] Price {price} rejected (<= 0)")
                return None
            
            # Fix common price interpretation issues for Bitcoin
            if asset == 'BTC':
                # Handle decimal notation (1.4, 1.7, etc. meaning 140k, 170k)
                # If price is between 1 and 10, it's likely decimal notation for hundreds of thousands
                if 1 <= price <= 10:
                    price = price * 100000
                    print(f"   [DEBUG] Normalized decimal notation: {price/100000:.1f} -> ${price:,.0f}")
                
                # Handle small numbers that are likely in thousands (e.g., "120" means 120k)
                elif price < 1000:
                    price = price * 1000
                    print(f"   [DEBUG] Normalized to thousands: {price/1000:.0f} -> ${price:,.0f}")
                
                # Handle numbers like 1700 that might be 170k (based on context like "1.70")
                elif 1000 <= price < 10000:
                    # Check if this looks like it came from decimal notation
                    # Numbers like 1400, 1700 likely came from "1.4", "1.7"
                    if price % 100 == 0 and 1000 <= price <= 3000:
                        price = price * 100
                        print(f"   [DEBUG] Normalized likely decimal: {price/100000:.1f} -> ${price:,.0f}")
                
                # Special case for teen numbers that might be 110k-190k
                elif 10 <= price <= 20:
                    price = (price + 100) * 1000
                    print(f"   [DEBUG] Normalized teen number: {(price/1000-100):.0f} -> ${price:,.0f}")
            
            # Get raw timeframe from Stage 2
            timeframe_str = pred_data.get('timeframe')
            if timeframe_str is None:
                timeframe_str = ''
            else:
                timeframe_str = str(timeframe_str).strip()
            
            # REJECT predictions without timeframes
            if not timeframe_str or timeframe_str.lower() in ['none', 'null', '', 'no timeframe']:
                print(f"   [DEBUG] Rejecting prediction - no timeframe: {asset} ${price}")
                return None
            
            # For now, store raw timeframe - Stage 3 will parse it
            timeframe_data = {
                'raw_timeframe': timeframe_str,
                'time_frame': None,  # Will be set by Stage 3
                'time_value': None,  # Will be set by Stage 3
                'predicted_date': None  # Will be set by Stage 3
            }
            
            timeframe_parsing_info = {
                'original_timeframe': timeframe_str,
                'episode_date': episode_info.get('date', datetime.now().isoformat()),
                'parsing_flow': 'deferred_to_stage3',
                'parsing_notes': ['Timeframe parsing deferred to Stage 3']
            }
            
            # Parse confidence
            confidence_str = pred_data.get('confidence', 'uncertain').lower()
            confidence_map = {
                'high': Confidence.HIGH,
                'medium': Confidence.MEDIUM,
                'low': Confidence.LOW,
                'uncertain': Confidence.UNCERTAIN
            }
            confidence = confidence_map.get(confidence_str, Confidence.UNCERTAIN)
            
            # Ensure price is float before creating prediction
            if not isinstance(price, (int, float)):
                print(f"   [DEBUG] Price type error: got {type(price)} instead of float")
                return None
                
            # Try to find exact quote location in segments for better timestamp
            quote_text = pred_data.get('quote', '')
            quote_match = None
            
            if quote_text and hasattr(self, 'current_segments') and self.current_segments:
                quote_match = self._find_quote_in_segments(quote_text, self.current_segments)
            
            # Use quote match timestamp if found, otherwise fall back to snippet timestamp
            if quote_match:
                timestamp_start = quote_match['start']
                formatted_timestamp = self._format_timestamp(timestamp_start)
                print(f"   [TIMESTAMP] Using quote match: {formatted_timestamp}")
            else:
                # Use snippet timestamp if available
                timestamp_start = pred_data.get('snippet_timestamp_start')
                if timestamp_start is not None:
                    # Format the timestamp
                    formatted_timestamp = self._format_timestamp(timestamp_start)
                else:
                    # Fallback to LLM-provided timestamp
                    formatted_timestamp = pred_data.get('timestamp', 'unknown')
            
            # Add final parsing results to debug info
            timeframe_parsing_info['final_time_frame'] = str(timeframe_data.get('time_frame'))
            timeframe_parsing_info['final_time_value'] = timeframe_data.get('time_value')
            timeframe_parsing_info['final_predicted_date'] = timeframe_data.get('predicted_date')
            if 'parsing_note' in timeframe_data:
                timeframe_parsing_info['parsing_notes'].append(timeframe_data['parsing_note'])
            
            # Create prediction
            prediction = Prediction(
                asset=asset,
                prediction_type=PredictionType.PRICE_TARGET,
                value=float(price),  # Explicitly cast to float
                predicted_date=timeframe_data.get('predicted_date'),
                time_frame=timeframe_data.get('time_frame'),
                time_value=timeframe_data.get('time_value'),
                confidence=confidence,
                reasoning=pred_data.get('reasoning'),
                episode=episode_info.get('title', ''),
                episode_date=episode_info.get('date', ''),
                timestamp=formatted_timestamp,
                timestamp_start=timestamp_start,
                timestamp_end=pred_data.get('snippet_timestamp_end'),
                raw_text=pred_data.get('quote', '')[:500],
                prediction_id=self._generate_id(asset, price, timeframe_data.get('predicted_date')),
                timeframe_parsing_info=timeframe_parsing_info
            )
            
            # Generate YouTube link if we have video ID and timestamp
            video_id = episode_info.get('video_id')
            if video_id and timestamp_start is not None:
                prediction.youtube_link = prediction.generate_youtube_link(video_id)
            
            return prediction
            
        except (ValueError, KeyError, TypeError) as e:
            return None
    
    def _parse_structured_timeframe(self, timeframe_str: str, timeframe_type: str, 
                                     timeframe_value: any, episode_date: str) -> Dict:
        """Parse structured timeframe data from LLM"""
        result = {}
        
        try:
            current_date = datetime.fromisoformat(episode_date.split('T')[0])
        except:
            current_date = datetime.now()
            
        if timeframe_type == 'specific_date':
            if isinstance(timeframe_value, str) and '-' in timeframe_value:
                # Already in YYYY-MM-DD format
                result['predicted_date'] = timeframe_value
                result['time_frame'] = TimeFrame.SPECIFIC_DATE
                result['parsing_note'] = f'Specific date already formatted: {timeframe_value}'
            else:
                # Need to parse "July 18th" style - use the original parser
                result = self._parse_timeframe(timeframe_str, episode_date)
                result['parsing_note'] = f'Specific date needed text parsing: {timeframe_str}'
                return result
                
        elif timeframe_type == 'days':
            result['time_frame'] = TimeFrame.DAYS
            result['time_value'] = int(timeframe_value) if timeframe_value else 7
            result['parsing_note'] = f'Days: {timeframe_value or "defaulted to 7"}'
            
        elif timeframe_type == 'weeks':
            result['time_frame'] = TimeFrame.WEEKS
            result['time_value'] = int(timeframe_value) if timeframe_value else 2
            result['parsing_note'] = f'Weeks: {timeframe_value or "defaulted to 2"}'
            
        elif timeframe_type == 'months':
            result['time_frame'] = TimeFrame.MONTHS
            result['time_value'] = int(timeframe_value) if timeframe_value else 3
            result['parsing_note'] = f'Months: {timeframe_value or "defaulted to 3"}'
            
        elif timeframe_type == 'conditional':
            # For conditional predictions, use a reasonable default
            result['time_frame'] = TimeFrame.MONTHS
            result['time_value'] = 2
            result['parsing_note'] = f'Conditional defaulted to 2 months: "{timeframe_str}"'
            
        elif timeframe_type == 'market_cycle':
            # Bull runs typically last 6-12 months
            result['time_frame'] = TimeFrame.MONTHS
            result['time_value'] = 6
            result['parsing_note'] = f'Market cycle defaulted to 6 months: "{timeframe_str}"'
            
        else:
            # Unknown type, fallback to parser
            result = self._parse_timeframe(timeframe_str, episode_date)
            result['parsing_note'] = f'Unknown type "{timeframe_type}", used text parser'
            return result
            
        return result
    
    def _parse_timeframe(self, timeframe_str: str, episode_date: str) -> Dict:
        """Parse timeframe string into structured data"""
        timeframe_lower = timeframe_str.lower()
        result = {}
        parsing_notes = []
        
        # Try to parse specific dates
        if any(month in timeframe_lower for month in [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'
        ]):
            for month_num, month_name in enumerate([
                'january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december'
            ], 1):
                if month_name in timeframe_lower:
                    import re
                    day_match = re.search(r'(\d{1,2})(?:st|nd|rd|th)?', timeframe_lower)
                    day = int(day_match.group(1)) if day_match else 1
                    
                    try:
                        current_date = datetime.fromisoformat(episode_date.split('T')[0])
                        parsing_notes.append(f"Episode date: {episode_date}")
                    except:
                        current_date = datetime.now()
                        parsing_notes.append(f"Failed to parse episode date '{episode_date}', using today")
                    
                    year = current_date.year
                    if month_num < current_date.month:
                        year += 1
                        parsing_notes.append(f"Month {month_num} < current {current_date.month}, using next year")
                    
                    predicted_date = f"{year}-{month_num:02d}-{day:02d}"
                    
                    # Sanity check: if the predicted date is before the episode date, add a year
                    try:
                        pred_dt = datetime.fromisoformat(predicted_date)
                        if pred_dt < current_date:
                            year += 1
                            predicted_date = f"{year}-{month_num:02d}-{day:02d}"
                            parsing_notes.append(f"Predicted date was in past, adjusted to next year")
                    except:
                        pass
                    
                    result['predicted_date'] = predicted_date
                    result['time_frame'] = TimeFrame.SPECIFIC_DATE
                    parsing_notes.append(f"Parsed month '{month_name}' → {predicted_date}")
                    break
        
        # Relative timeframes
        elif 'tomorrow' in timeframe_lower:
            try:
                current_date = datetime.fromisoformat(episode_date)
            except:
                current_date = datetime.now()
            next_day = current_date.replace(day=current_date.day + 1)
            result['predicted_date'] = next_day.isoformat()[:10]
            result['time_frame'] = TimeFrame.SPECIFIC_DATE
            parsing_notes.append(f"Parsed 'tomorrow' → {result['predicted_date']}")
        
        elif 'next week' in timeframe_lower or 'by next week' in timeframe_lower:
            result['time_frame'] = TimeFrame.WEEKS
            result['time_value'] = 1
        
        elif 'end of year' in timeframe_lower or 'eoy' in timeframe_lower or 'this year' in timeframe_lower or 'by year' in timeframe_lower:
            result['time_frame'] = TimeFrame.EOY
            try:
                current_date = datetime.fromisoformat(episode_date)
            except:
                current_date = datetime.now()
            result['predicted_date'] = f"{current_date.year}-12-31"
            parsing_notes.append(f"Parsed 'end of year' → {result['predicted_date']}")
        
        elif 'month' in timeframe_lower:
            import re
            num_match = re.search(r'(\d+)\s*month', timeframe_lower)
            if num_match:
                result['time_frame'] = TimeFrame.MONTHS
                result['time_value'] = int(num_match.group(1))
            elif 'next month' in timeframe_lower:
                result['time_frame'] = TimeFrame.MONTHS
                result['time_value'] = 1
        
        elif 'week' in timeframe_lower:
            import re
            num_match = re.search(r'(\d+)\s*week', timeframe_lower)
            if num_match:
                result['time_frame'] = TimeFrame.WEEKS
                result['time_value'] = int(num_match.group(1))
            elif 'this week' in timeframe_lower:
                result['time_frame'] = TimeFrame.WEEKS
                result['time_value'] = 0
        
        elif 'day' in timeframe_lower:
            import re
            num_match = re.search(r'(\d+)\s*day', timeframe_lower)
            if num_match:
                result['time_frame'] = TimeFrame.DAYS
                result['time_value'] = int(num_match.group(1))
        
        # Handle specific contextual timeframes
        elif 'imminently' in timeframe_lower or 'any day' in timeframe_lower:
            result['time_frame'] = TimeFrame.DAYS
            result['time_value'] = 3  # Very near term
            
        elif 'very soon' in timeframe_lower or 'right now' in timeframe_lower:
            result['time_frame'] = TimeFrame.DAYS
            result['time_value'] = 7  # Within a week
            
        elif 'near-term' in timeframe_lower or 'short term' in timeframe_lower:
            result['time_frame'] = TimeFrame.WEEKS
            result['time_value'] = 2  # Couple weeks
            
        elif 'bull run' in timeframe_lower or 'this cycle' in timeframe_lower:
            result['time_frame'] = TimeFrame.MONTHS
            result['time_value'] = 6  # Bull cycles typically last months
            
        elif 'after' in timeframe_lower or 'once' in timeframe_lower or 'when' in timeframe_lower:
            # Conditional - more varied based on context
            if 'tomorrow' in timeframe_lower:
                result['time_frame'] = TimeFrame.DAYS
                result['time_value'] = 1
            elif 'week' in timeframe_lower:
                result['time_frame'] = TimeFrame.WEEKS
                result['time_value'] = 1
            else:
                result['time_frame'] = TimeFrame.MONTHS
                result['time_value'] = 2  # Conditional events typically happen in months
            
        # Handle market momentum context
        elif 'momentum' in timeframe_lower or 'current' in timeframe_lower:
            result['time_frame'] = TimeFrame.WEEKS
            result['time_value'] = 2  # Current momentum plays out in weeks
        
        # If no specific timeframe matched but there's text, vary by context
        elif timeframe_str:
            # Don't always default to same values
            text_len = len(timeframe_str)
            if text_len < 10:  # Very short like "soon"
                result['time_frame'] = TimeFrame.DAYS
                result['time_value'] = 14
                parsing_notes.append(f"Defaulted to DAYS/14 based on text length ({text_len} chars)")
            elif text_len < 25:  # Medium phrases
                result['time_frame'] = TimeFrame.WEEKS
                result['time_value'] = 3
                parsing_notes.append(f"Defaulted to WEEKS/3 based on text length ({text_len} chars)")
            else:  # Longer conditional phrases
                result['time_frame'] = TimeFrame.MONTHS  
                result['time_value'] = 1
                parsing_notes.append(f"Defaulted to MONTHS/1 based on text length ({text_len} chars)")
        
        # Add all parsing notes to result
        if parsing_notes:
            result['parsing_note'] = "; ".join(parsing_notes)
        
        return result
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to readable timestamp (HH:MM:SS or MM:SS)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def _generate_id(self, asset: str, value: float, date: Optional[str]) -> str:
        """Generate unique prediction ID"""
        import hashlib
        data = f"{asset}_{value}_{date}_{datetime.now().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
    
    def _get_stage2_system_prompt(self) -> str:
        """Get system prompt for Stage 2 prediction extraction"""
        return """You are extracting cryptocurrency price predictions from podcast snippets.

For each snippet, identify:
1. The speaker making the prediction (if identifiable)
2. The asset being predicted (BTC, ETH, MSTR, etc.)
3. The target price or percentage
4. The timeframe - COPY EXACTLY what the speaker said about timing (e.g. "next week", "by end of year", "when we break 100k")
5. Confidence level (based on how they phrase it)
6. Any conditions mentioned

Be precise and only extract predictions made BY THE SPEAKERS, not discussions about others' predictions.

ASSET MAPPING:
- "misty" → MSTY
- "similar", "silver", "sampler", "summer" → SMLR  
- "mst" → MSTR
- "big coin" → BTC
- "meta planet" → METAPLANET
- "I bet", "IBIT" → IBIT
- "galaxy" → GLXY

IMPORTANT: For timeframe, just copy the exact words used. Do NOT parse, categorize, or interpret. Examples:
- If they say "next Tuesday" → timeframe: "next Tuesday"
- If they say "by the end of this cycle" → timeframe: "by the end of this cycle"
- If they say "tomorrow" → timeframe: "tomorrow"
- If no timeframe mentioned → timeframe: ""

Output JSON:
{
  "predictions": [
    {
      "speaker": "Speaker name or Unknown",
      "asset": "BTC",
      "price": 150000,
      "timeframe": "end of 2025",
      "confidence": "high/medium/low",
      "conditions": "if ETF approval happens",
      "reasoning": "their stated logic",
      "quote": "exact quote from transcript showing the prediction"
    }
  ]
}"""
    
    def _stage3_refinement(self, predictions: List[Prediction], episode_info: Dict) -> List[Prediction]:
        """Stage 3: Deep refinement with expensive model or dry run"""
        print(f"\n[STAGE 3 REFINEMENT] Processing {len(predictions)} predictions...")
        
        if self.stage3_dry_run:
            print("[DRY RUN MODE] Logging what would be sent to Stage 3...")
            
            # Create the prompt that would be sent
            system_prompt = self._get_stage3_system_prompt()
            
            # Convert predictions to dict format for the prompt
            pred_dicts = []
            for pred in predictions:
                pred_dicts.append({
                    'asset': pred.asset,
                    'value': pred.value,
                    'timeframe': str(pred.time_frame.value) if pred.time_frame else 'unknown',
                    'confidence': pred.confidence.value,
                    'reasoning': pred.reasoning,
                    'quote': pred.raw_text,
                    'timestamp': pred.timestamp
                })
            
            user_prompt = f"""Episode: {episode_info.get('title', '')}
Date: {episode_info.get('date', '')}

Predictions to refine:
{json.dumps(pred_dicts, indent=2)}

Please perform deep analysis and refinement of these predictions."""
            
            # Calculate tokens
            input_tokens = self.count_tokens(system_prompt + user_prompt, self.stage3_model)
            self.token_usage['refinement']['input'] += input_tokens
            self.token_usage['refinement']['calls'] += 1
            
            # Log the dry run
            dry_run_entry = {
                'timestamp': datetime.now().isoformat(),
                'episode': episode_info.get('title', ''),
                'predictions_count': len(predictions),
                'model': self.stage3_model,
                'estimated_input_tokens': input_tokens,
                'estimated_cost': (input_tokens / 1_000_000) * 10.0,  # Assuming $10/1M for expensive model
                'system_prompt_preview': system_prompt[:200] + '...',
                'user_prompt_preview': user_prompt[:500] + '...',
                'full_prompt_length': len(system_prompt + user_prompt)
            }
            
            self.stage3_dry_run_log.append(dry_run_entry)
            
            # Save dry run log to file
            dry_run_file = self.debug_log_dir / f"stage3_dry_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(dry_run_file, 'w') as f:
                json.dump(dry_run_entry, f, indent=2)
            
            # Also log full details to main debug log
            self._log_debug("stage3_dry_run", dry_run_entry)
            
            print(f"[DRY RUN] Would send {input_tokens:,} tokens to {self.stage3_model}")
            print(f"[DRY RUN] Estimated cost: ${(input_tokens / 1_000_000) * 10.0:.4f}")
            print(f"[DRY RUN] Log saved to: {dry_run_file}")
            
            # Return predictions unchanged
            return predictions
        
        else:
            # Actual Stage 3 processing
            if not self.refinement_client:
                print("[ERROR] No refinement client available")
                return predictions
            
            print("[STAGE 3] Starting actual refinement...")
            
            # Create Stage 3 request
            system_prompt = self._get_stage3_system_prompt()
            user_prompt = self._create_stage3_prompt(predictions, episode_info)
            
            # Log full request
            stage3_request = {
                'timestamp': datetime.now().isoformat(),
                'episode': episode_info.get('title', ''),
                'predictions_count': len(predictions),
                'model': self.stage3_model,
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'predictions_sent': [{
                    'asset': p.asset,
                    'value': p.value,
                    'raw_timeframe': p.timeframe_parsing_info.get('original_timeframe', '') if hasattr(p, 'timeframe_parsing_info') and p.timeframe_parsing_info else '',
                    'confidence': p.confidence.value,
                    'prediction_id': p.prediction_id
                } for p in predictions]
            }
            
            # Save request log
            request_file = self.debug_log_dir / f"stage3_request_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(request_file, 'w') as f:
                json.dump(stage3_request, f, indent=2)
            print(f"[STAGE 3] Request saved to: {request_file}")
            
            # Log to main debug log
            self._log_debug("stage3_request", stage3_request)
            
            try:
                # Use specialized methods if available
                if hasattr(self.refinement_client, 'deep_extract_predictions'):
                    # For GPT4TurboClient - convert predictions to serializable format
                    pred_data = []
                    for p in predictions:
                        pred_dict = {
                            'asset': p.asset,
                            'value': p.value,
                            'original_timeframe': p.timeframe_parsing_info.get('original_timeframe', '') if hasattr(p, 'timeframe_parsing_info') and p.timeframe_parsing_info else '',
                            'confidence': p.confidence.value if hasattr(p.confidence, 'value') else str(p.confidence),
                            'quote': p.raw_text,
                            'timestamp': p.timestamp,
                            'episode_date': p.episode_date,
                            'prediction_id': p.prediction_id,
                            'reasoning': p.reasoning
                        }
                        pred_data.append(pred_dict)
                    
                    refined_data = self.refinement_client.deep_extract_predictions(
                        json.dumps(pred_data),
                        episode_info
                    )
                elif hasattr(self.refinement_client, 'process_entire_transcript'):
                    # For GPT41Client - would need full transcript
                    print("[INFO] GPT-4-1 client would benefit from full transcript")
                    refined_data = predictions  # Keep original for now
                else:
                    # Generic refinement - already have prompts from above
                    print(f"[STAGE 3] Sending {len(user_prompt)} chars to {self.stage3_model}...")
                    
                    # Track time
                    start_time = time.time()
                    
                    response = self.refinement_client.extract_for_stage(
                        stage=ExtractionStage.DETAILED_EXTRACT,
                        text=user_prompt,
                        system_prompt=system_prompt,
                        temperature=0.0,
                        parse_json=True
                    )
                    
                    elapsed = time.time() - start_time
                    print(f"[STAGE 3] Response received in {elapsed:.1f}s")
                    
                    # Log raw response
                    stage3_response = {
                        'timestamp': datetime.now().isoformat(),
                        'elapsed_seconds': elapsed,
                        'raw_response': response,
                        'response_type': type(response).__name__
                    }
                    
                    response_file = self.debug_log_dir / f"stage3_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(response_file, 'w') as f:
                        json.dump(stage3_response, f, indent=2)
                    print(f"[STAGE 3] Response saved to: {response_file}")
                    
                    # Log to main debug log
                    self._log_debug("stage3_response", stage3_response)
                    
                    refined_data = response.get('predictions', []) if isinstance(response, dict) else []
                    print(f"[STAGE 3] Extracted {len(refined_data)} refined predictions")
                
                # Token tracking is handled automatically by refinement_client.extract_for_stage()
                # Get the actual token usage from the client
                if hasattr(self.refinement_client, 'token_usage'):
                    client_usage = self.refinement_client.token_usage.get(ExtractionStage.DETAILED_EXTRACT, {})
                    self.token_usage['refinement']['input'] = client_usage.get('input', 0)
                    self.token_usage['refinement']['output'] = client_usage.get('output', 0)
                    self.token_usage['refinement']['calls'] = client_usage.get('calls', 0)
                
                print(f"[STAGE 3] Refinement complete")
                
                # Merge refined timeframe data back into original predictions
                if refined_data and isinstance(refined_data, list):
                    # Create a map of prediction_id to refined data
                    refined_map = {}
                    for refined in refined_data:
                        if isinstance(refined, dict) and 'prediction_id' in refined:
                            refined_map[refined['prediction_id']] = refined
                    
                    # Update original predictions with refined timeframe data
                    for pred in predictions:
                        if pred.prediction_id in refined_map:
                            refined = refined_map[pred.prediction_id]
                            
                            # Update timeframe parsing info
                            if 'parsed_timeframe' in refined:
                                if not hasattr(pred, 'timeframe_parsing_info'):
                                    pred.timeframe_parsing_info = {}
                                
                                parsed = refined['parsed_timeframe']
                                if isinstance(parsed, dict):
                                    # Update TimeFrame enum if provided
                                    if 'time_frame' in parsed:
                                        try:
                                            pred.time_frame = TimeFrame(parsed['time_frame'])
                                        except:
                                            pred.time_frame = TimeFrame.NONE
                                    
                                    # Update time value
                                    if 'time_value' in parsed:
                                        pred.time_value = parsed['time_value']
                                    
                                    # Update predicted date
                                    if 'predicted_date' in parsed:
                                        pred.predicted_date = parsed['predicted_date']
                                    
                                    # Update parsing info
                                    pred.timeframe_parsing_info.update({
                                        'stage3_parsed': True,
                                        'parsed_timeframe': parsed,
                                        'parsing_notes': parsed.get('notes', [])
                                    })
                                    
                                    print(f"   [STAGE 3] Updated timeframe for {pred.asset} ${pred.value}: {parsed}")
                
                return predictions
                    
            except Exception as e:
                print(f"[ERROR] Stage 3 refinement failed: {e}")
                return predictions
    
    def _get_stage3_system_prompt(self) -> str:
        """Get system prompt for Stage 3 deep refinement"""
        return """You are performing deep analysis and refinement of cryptocurrency predictions.

Your PRIMARY task is to parse and interpret timeframes:
1. Convert raw timeframe strings to structured dates/periods
2. Handle context-dependent timing (e.g., "next week" relative to episode date)
3. Identify conditional vs specific timeframes
4. Ensure predictions are in the future (not past dates)

For each prediction, you MUST:
1. Parse the raw timeframe string into:
   - predicted_date: YYYY-MM-DD format if specific
   - time_frame_type: "days", "weeks", "months", "years", "specific_date", "conditional"
   - time_value: numeric value (e.g., 2 for "2 weeks")
   
2. Examples:
   - "next week" → time_frame_type: "weeks", time_value: 1
   - "by December" → predicted_date: "2025-12-01", time_frame_type: "specific_date"
   - "when we hit 100k" → time_frame_type: "conditional", note: "dependent on BTC reaching 100k"
   - "end of year" → predicted_date: "2025-12-31", time_frame_type: "specific_date"

3. Also enhance:
   - Validate prediction logic
   - Add market context
   - Flag suspicious predictions

Output JSON with parsed timeframes and enhanced metadata."""
    
    def _create_stage3_prompt(self, predictions: List[Prediction], episode_info: Dict) -> str:
        """Create user prompt for Stage 3"""
        pred_data = []
        for p in predictions:
            # Get the raw timeframe from parsing info
            raw_timeframe = ''
            if hasattr(p, 'timeframe_parsing_info') and p.timeframe_parsing_info:
                raw_timeframe = p.timeframe_parsing_info.get('original_timeframe', '')
            
            pred_data.append({
                'asset': p.asset,
                'value': p.value,
                'raw_timeframe': raw_timeframe,
                'confidence': p.confidence.value,
                'reasoning': p.reasoning,
                'quote': p.raw_text,
                'prediction_id': p.prediction_id
            })
        
        return f"""Episode: {episode_info.get('title', '')}
Episode Date: {episode_info.get('date', '')}

IMPORTANT: Parse the raw_timeframe strings relative to the episode date above.

Predictions to analyze and refine:
{json.dumps(pred_data, indent=2)}

For each prediction:
1. Parse the raw_timeframe into structured format
2. Ensure dates are in the future relative to episode date
3. Return the enhanced predictions with parsed timeframes."""