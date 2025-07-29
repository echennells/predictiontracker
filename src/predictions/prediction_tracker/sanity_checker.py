"""
Sanity checker for predictions using LLM
"""
from typing import List, Dict, Optional
import json
from datetime import datetime

from llm.llm_client import OpenAIClient
from .models import Prediction


class PredictionSanityChecker:
    """Use LLM to sanity check and fix obvious issues in predictions"""
    
    def __init__(self, debug_logger=None):
        self.client = OpenAIClient()
        # Use the expensive model for high quality checking
        self.client.model = "gpt-4o"
        self.debug_logger = debug_logger
    
    def check_predictions(self, predictions: List[Prediction], episode_info: Dict) -> List[Prediction]:
        """
        Run sanity checks on predictions and fix obvious issues
        """
        if not predictions:
            return predictions
        
        print(f"\n[SANITY CHECK] Checking {len(predictions)} predictions with {self.client.model}...")
        
        # Convert predictions to simple format for LLM
        predictions_data = []
        for pred in predictions:
            predictions_data.append({
                'asset': pred.asset,
                'price': pred.value,
                'quote': pred.raw_text,
                'timeframe': pred.time_frame.value if pred.time_frame else None,
                'timestamp': pred.timestamp,
                'episode': pred.episode
            })
        
        system_prompt = """You are a sanity checker for crypto predictions extracted from podcasts.

Your job is to identify and fix OBVIOUS errors in the predictions.

COMMON ISSUES TO FIX:
1. Bitcoin prices that make no sense (e.g., $1,700 when they meant $170,000)
   - In Bitcoin discussions, "1.7" usually means $170,000
   - Small numbers like 120 usually mean $120,000
   
2. Asset names that are clearly wrong:
   - "similar cel" → "SMLR" (Semler Scientific)
   - "summer" → "SMLR" 
   - "misty" → "MSTY" (MicroStrategy ETF)
   
3. Missing or wrong timestamps:
   - If timestamp is "00:00:00" or missing, flag it
   
4. Timeframes that don't make sense:
   - If discussing current momentum but timeframe is years away

For each prediction, return:
- fixed_asset: Corrected asset symbol if needed
- fixed_price: Corrected price if obviously wrong
- confidence_in_fix: "high" if obvious fix, "medium" if likely, "low" if unsure
- issue_found: Brief description of what was wrong
- keep_prediction: true/false - whether to keep this prediction

Be conservative - only fix OBVIOUS errors. When in doubt, leave it as is."""

        user_prompt = f"""Review these predictions from episode: {episode_info.get('title', 'Unknown')}

{json.dumps(predictions_data, indent=2)}

For each prediction, identify any obvious errors and suggest fixes.
Return a JSON array with one entry per prediction."""

        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            content = response['choices'][0]['message']['content']
            
            # Parse response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            # Handle different response formats
            if isinstance(result, dict):
                fixes = result.get('predictions', result.get('fixes', []))
            elif isinstance(result, list):
                fixes = result
            else:
                print(f"[SANITY CHECK] Unexpected response format: {type(result)}")
                fixes = []
            
            # Ensure we have the right number of fixes
            if len(fixes) != len(predictions):
                print(f"[SANITY CHECK] Warning: Got {len(fixes)} fixes for {len(predictions)} predictions")
                # Pad with empty dicts if needed
                while len(fixes) < len(predictions):
                    fixes.append({})
            
            # Apply fixes
            fixed_predictions = []
            sanity_results = []
            
            for i, (pred, fix) in enumerate(zip(predictions, fixes)):
                # Ensure fix is a dictionary
                if not isinstance(fix, dict):
                    print(f"   ⚠️  Warning: Invalid fix format for prediction {i+1}, skipping sanity check")
                    fixed_predictions.append(pred)
                    continue
                    
                sanity_result = {
                    'original_asset': pred.asset,
                    'original_price': pred.value,
                    'quote': pred.raw_text[:100] if pred.raw_text else None,
                    'fix_data': fix,
                    'action': None,
                    'changes': []
                }
                
                if not fix.get('keep_prediction', True):
                    print(f"   ✗ Removed: {pred.asset} ${pred.value:,.0f} - {fix.get('issue_found', 'Failed sanity check')}")
                    sanity_result['action'] = 'rejected'
                    sanity_result['reason'] = fix.get('issue_found', 'Failed sanity check')
                    sanity_results.append(sanity_result)
                    continue
                
                # Apply fixes if confidence is high enough
                if fix.get('confidence_in_fix') in ['high', 'medium']:
                    if fix.get('fixed_asset') and fix['fixed_asset'] != pred.asset:
                        print(f"   📝 Fixed asset: {pred.asset} → {fix['fixed_asset']}")
                        sanity_result['changes'].append({
                            'field': 'asset',
                            'from': pred.asset,
                            'to': fix['fixed_asset']
                        })
                        pred.asset = fix['fixed_asset']
                    
                    if fix.get('fixed_price') and fix['fixed_price'] != pred.value:
                        print(f"   📝 Fixed price: ${pred.value:,.0f} → ${fix['fixed_price']:,.0f}")
                        sanity_result['changes'].append({
                            'field': 'price',
                            'from': pred.value,
                            'to': fix['fixed_price']
                        })
                        pred.value = fix['fixed_price']
                
                if fix.get('issue_found'):
                    print(f"   ⚠️  Issue: {fix['issue_found']}")
                    sanity_result['issue'] = fix['issue_found']
                
                sanity_result['action'] = 'fixed' if sanity_result['changes'] else 'passed'
                sanity_results.append(sanity_result)
                fixed_predictions.append(pred)
            
            # Log debug data if logger available
            if self.debug_logger:
                self.debug_logger("sanity_check", {
                    "total_checked": len(predictions),
                    "total_passed": len(fixed_predictions),
                    "total_rejected": len(predictions) - len(fixed_predictions),
                    "model_used": self.client.model,
                    "results": sanity_results
                })
            
            print(f"[SANITY CHECK] Complete: {len(fixed_predictions)} predictions passed")
            return fixed_predictions
            
        except Exception as e:
            print(f"[SANITY CHECK] Error: {e}")
            # If sanity check fails, return original predictions
            return predictions