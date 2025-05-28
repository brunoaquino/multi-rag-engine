#!/usr/bin/env python3
"""
Script to register RAG pipeline with Hayhooks
"""

import requests
import json
import time
import sys

def register_rag_pipeline():
    """Register a simple RAG pipeline with Hayhooks"""
    
    # Pipeline definition for a basic embedding pipeline
    pipeline_yaml = """
version: ignore
components:
  - name: embedder
    type: OpenAITextEmbedder
    params:
      model: text-embedding-ada-002
connections: []
"""
    
    # Try to register pipeline
    try:
        # First check if Hayhooks is running
        response = requests.get("http://localhost:1416/status", timeout=5)
        if response.status_code != 200:
            print("❌ Hayhooks not responding")
            return False
        
        print("✅ Hayhooks is running")
        
        # Register a simple pipeline
        headers = {"Content-Type": "application/x-yaml"}
        
        # Register rag_pipeline 
        response = requests.post(
            "http://localhost:1416/rag_pipeline", 
            data=pipeline_yaml,
            headers=headers
        )
        
        if response.status_code in [200, 201]:
            print("✅ RAG pipeline registered successfully!")
            return True
        else:
            print(f"❌ Failed to register pipeline: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error registering pipeline: {e}")
        return False

def create_simple_embedding_endpoint():
    """Create a simple embedding endpoint that works with our integration"""
    
    # Create a minimal embedding pipeline using direct API calls
    pipeline_config = {
        "name": "rag_pipeline",
        "description": "Simple RAG pipeline for document processing",
        "components": {
            "embedder": {
                "type": "OpenAITextEmbedder",
                "params": {
                    "model": "text-embedding-ada-002"
                }
            }
        },
        "connections": []
    }
    
    try:
        # Try to POST the pipeline configuration
        response = requests.post(
            "http://localhost:1416/rag_pipeline",
            json=pipeline_config,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
        
        if response.status_code in [200, 201]:
            print("✅ Pipeline registered via JSON!")
            return True
        else:
            print(f"❌ JSON registration failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error with JSON registration: {e}")
        return False

def main():
    print("🚀 Registering RAG pipeline with Hayhooks...")
    
    # Wait for Hayhooks to be ready
    print("⏳ Waiting for Hayhooks...")
    time.sleep(3)
    
    # Try YAML registration
    if register_rag_pipeline():
        print("✅ Pipeline registration completed!")
        return True
    
    print("⚠️ YAML registration failed, trying JSON...")
    
    # Try JSON registration  
    if create_simple_embedding_endpoint():
        print("✅ JSON Pipeline registration completed!")
        return True
    
    print("❌ All registration methods failed")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 