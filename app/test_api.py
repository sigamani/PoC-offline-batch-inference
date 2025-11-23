#!/usr/bin/env python3
"""
Simple test client to verify API connectivity
"""
import requests
import json

def test_api(server_url):
    """Test API endpoints"""
    
    # Test health endpoint
    try:
        print(f"🔍 Testing health endpoint: {server_url}/health")
        response = requests.get(f"{server_url}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print("✅ Health check successful:")
            print(f"   Status: {health.get('status')}")
            print(f"   Worker: {health.get('worker_id')}")
            print(f"   Model: {health.get('model')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test single generation
    try:
        print(f"\n🚀 Testing single generation...")
        payload = {
            "prompt": "What is artificial intelligence?",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(f"{server_url}/generate", json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print("✅ Single generation successful:")
            print(f"   Text: {result['text'][:100]}...")
            print(f"   Tokens: {result['tokens_generated']}")
            print(f"   Time: {result['inference_time']:.2f}s")
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False
    
    # Test batch generation
    try:
        print(f"\n📦 Testing batch generation...")
        payload = {
            "prompts": [
                "What is machine learning?",
                "Explain neural networks."
            ],
            "max_tokens": 30,
            "temperature": 0.7
        }
        
        response = requests.post(f"{server_url}/generate_batch", json=payload, timeout=120)
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch generation successful:")
            print(f"   Total time: {result['total_time']:.2f}s")
            for i, res in enumerate(result['results']):
                print(f"   {i+1}. {res['text'][:50]}... ({res['tokens_generated']} tokens)")
        else:
            print(f"❌ Batch generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch generation error: {e}")
        return False
    
    print(f"\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python test_api.py <server_url>")
        print("Example: python test_api.py http://192.168.1.100:8000")
        sys.exit(1)
    
    server_url = sys.argv[1]
    test_api(server_url)