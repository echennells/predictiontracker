#!/usr/bin/env python3
"""
Test concurrent processing in GPT4OClient
"""
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.gpt4o_client import GPT4OClient
from config.config import OPENAI_API_KEY

def test_concurrent_processing():
    """Test concurrent vs sequential processing"""
    
    # Create client
    client = GPT4OClient(OPENAI_API_KEY)
    
    # Create dummy snippets
    snippets = []
    for i in range(10):
        snippets.append({
            'text': f"""
            Speaker {i}: I think Bitcoin is going to hit ${100000 + i*10000} by the end of next year.
            This is based on the current market trends and adoption rates.
            """
        })
    
    episode_info = {
        'title': 'Test Episode',
        'date': '2025-07-29'
    }
    
    # Test with concurrent processing
    print("Testing CONCURRENT processing...")
    os.environ['ENABLE_CONCURRENT_PROCESSING'] = 'true'
    os.environ['CONCURRENT_WORKERS'] = '3'
    
    start_time = time.time()
    predictions_concurrent = client.extract_predictions(snippets, episode_info)
    concurrent_time = time.time() - start_time
    
    print(f"\nConcurrent processing:")
    print(f"- Time: {concurrent_time:.1f}s")
    print(f"- Predictions found: {len(predictions_concurrent)}")
    
    # Test with sequential processing
    print("\n" + "="*50)
    print("Testing SEQUENTIAL processing...")
    os.environ['ENABLE_CONCURRENT_PROCESSING'] = 'false'
    
    start_time = time.time()
    predictions_sequential = client.extract_predictions(snippets, episode_info)
    sequential_time = time.time() - start_time
    
    print(f"\nSequential processing:")
    print(f"- Time: {sequential_time:.1f}s")
    print(f"- Predictions found: {len(predictions_sequential)}")
    
    # Compare results
    print(f"\n" + "="*50)
    print(f"RESULTS:")
    print(f"- Speedup: {sequential_time/concurrent_time:.1f}x")
    print(f"- Same number of predictions: {len(predictions_concurrent) == len(predictions_sequential)}")
    
    # Show a sample prediction
    if predictions_concurrent:
        print(f"\nSample prediction:")
        pred = predictions_concurrent[0]
        print(f"- Asset: {pred.get('asset')}")
        print(f"- Price: ${pred.get('price'):,}")
        print(f"- Timeframe: {pred.get('timeframe')}")

if __name__ == "__main__":
    test_concurrent_processing()