#!/usr/bin/env python3
# =================================================================
# Pipeline Initialization Script for Hayhooks
# =================================================================
"""
Initialization script to automatically register pipelines in Hayhooks.
This script runs when the container starts to set up the chat pipeline.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add pipelines to path
sys.path.insert(0, '/app/pipelines')

def setup_logging():
    """Configure logging for the initialization script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def wait_for_hayhooks_server():
    """Wait for Hayhooks server to be ready."""
    import urllib.request
    import urllib.error
    
    logger = logging.getLogger(__name__)
    max_retries = 30
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = urllib.request.urlopen('http://localhost:1416/status')
            if response.getcode() == 200:
                logger.info("Hayhooks server is ready!")
                return True
        except urllib.error.URLError:
            logger.info(f"Waiting for Hayhooks server... (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
    
    logger.error("Hayhooks server did not start within the expected time")
    return False

def register_pipeline_programmatically():
    """Register the chat pipeline programmatically using internal APIs."""
    logger = logging.getLogger(__name__)
    
    try:
        # Import Hayhooks internals
        from hayhooks.server.application import create_application
        from hayhooks.server.pipelines import PipelineRegistry
        from pipeline_wrapper import PipelineWrapper
        
        logger.info("Attempting to register chat pipeline programmatically...")
        
        # Get the pipeline registry
        registry = PipelineRegistry()
        
        # Create wrapper instance
        wrapper = PipelineWrapper()
        wrapper.setup()
        
        # Register the pipeline
        registry.add_pipeline("chat_pipeline", wrapper)
        
        logger.info("✅ Chat pipeline registered successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register pipeline programmatically: {e}")
        return False

def test_pipelines():
    """Test the pipeline functionality."""
    logger = logging.getLogger(__name__)
    
    # Test chat pipeline
    try:
        from chat_pipeline import create_chat_pipeline
        
        logger.info("Testing chat pipeline creation...")
        chat_pipeline = create_chat_pipeline()
        
        if chat_pipeline:
            logger.info("✅ Chat pipeline created successfully!")
            logger.info(f"Chat pipeline components: {list(chat_pipeline.graph.nodes.keys())}")
        else:
            logger.error("❌ Failed to create chat pipeline")
            return False
            
    except Exception as e:
        logger.error(f"Error testing chat pipeline: {e}")
        return False
    
    # Test RAG pipeline (optional)
    try:
        from rag_pipeline import create_rag_pipeline
        
        # Check if required environment variables are set
        required_env_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"⚠️  RAG pipeline test skipped - missing environment variables: {missing_vars}")
        else:
            logger.info("Testing RAG pipeline creation...")
            rag_pipeline = create_rag_pipeline(
                pinecone_index="test-haystack-rag",
                primary_llm="gpt-4o-mini"
            )
            
            if rag_pipeline:
                logger.info("✅ RAG pipeline created successfully!")
            else:
                logger.warning("⚠️  RAG pipeline creation returned None")
            
    except ImportError as e:
        logger.warning(f"⚠️  RAG pipeline not available - missing dependencies: {e}")
    except Exception as e:
        logger.warning(f"⚠️  RAG pipeline test failed: {e}")
    
    return True

def main():
    """Main initialization function."""
    logger = setup_logging()
    logger.info("🚀 Starting Hayhooks pipeline initialization...")
    
    # Test pipelines first
    if not test_pipelines():
        logger.error("Pipeline test failed, aborting initialization")
        return False
    
    # Wait for server to be ready
    if not wait_for_hayhooks_server():
        logger.error("Server not ready, aborting initialization")
        return False
    
    # Try programmatic registration
    if register_pipeline_programmatically():
        logger.info("✅ Pipeline initialization completed successfully!")
        return True
    else:
        logger.warning("Programmatic registration failed, but pipeline files are available")
        logger.info("Pipeline can still be used via direct API calls")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 