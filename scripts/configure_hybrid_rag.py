#!/usr/bin/env python3
"""
Configuration script to set up Hybrid RAG mode by default

This script demonstrates how to configure the RAG pipeline to use hybrid mode,
which combines document information with general knowledge for more complete answers.
"""

import sys
import os

# Add the pipelines directory to the path
sys.path.insert(0, 'hayhooks/pipelines')
sys.path.insert(0, 'pipelines')

from rag_pipeline import create_rag_pipeline, RAGConfig

def create_hybrid_rag_config():
    """Create a RAG configuration optimized for hybrid mode"""
    
    config = RAGConfig(
        # Basic settings
        pinecone_index="haystack-rag",
        primary_llm="gpt-4o-mini",
        embedding_model="text-embedding-ada-002",
        
        # Set hybrid mode as default
        rag_mode="hybrid",
        
        # Optimize for better responses
        max_tokens=1500,  # Allow longer responses for comprehensive answers
        temperature=0.7,  # Balance creativity with accuracy
        
        # Retrieval settings for better context
        top_k=5,  # Get more context documents
        chunk_size=512,
        chunk_overlap=50,
        
        # Enable caching for performance
        enable_cache=True,
        cache_query_results=True,
        cache_embeddings=True
    )
    
    return config

def test_hybrid_configuration():
    """Test the hybrid configuration with sample queries"""
    
    print("🚀 **TESTING HYBRID RAG CONFIGURATION**")
    print("=" * 50)
    
    # Create pipeline with hybrid config
    config = create_hybrid_rag_config()
    
    print(f"✅ Configuration created:")
    print(f"   - RAG Mode: {config.rag_mode}")
    print(f"   - Primary LLM: {config.primary_llm}")
    print(f"   - Max Tokens: {config.max_tokens}")
    print(f"   - Temperature: {config.temperature}")
    print(f"   - Top K: {config.top_k}")
    print(f"   - Cache Enabled: {config.enable_cache}")
    
    try:
        # Initialize pipeline
        rag_pipeline = create_rag_pipeline(
            pinecone_index=config.pinecone_index,
            primary_llm=config.primary_llm,
            rag_mode=config.rag_mode,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            enable_cache=config.enable_cache
        )
        
        print("\n✅ Hybrid RAG pipeline created successfully!")
        print("   The pipeline is now configured to:")
        print("   - Use document context as primary source")
        print("   - Enhance with general knowledge when helpful")
        print("   - Provide comprehensive, well-attributed answers")
        
        return rag_pipeline
        
    except Exception as e:
        print(f"\n❌ Error creating pipeline: {e}")
        print("   Make sure environment variables are set:")
        print("   - OPENAI_API_KEY")
        print("   - PINECONE_API_KEY")
        return None

def show_prompt_templates():
    """Show the different prompt templates available"""
    
    print("\n📝 **AVAILABLE RAG MODES AND TEMPLATES**")
    print("=" * 60)
    
    config = RAGConfig()
    
    print("🔹 **STRICT MODE** (Document-only):")
    print("   - Uses ONLY information from indexed documents")
    print("   - Responds with 'I cannot answer' if info not in docs")
    print("   - Best for: Compliance, legal, specific documentation queries")
    
    print("\n🔹 **HYBRID MODE** (Recommended):")
    print("   - Starts with document information")
    print("   - Enhances with general knowledge when helpful")
    print("   - Clearly distinguishes between sources")
    print("   - Best for: General Q&A, educational content, comprehensive answers")
    
    print("\n🔹 **ENHANCED MODE** (Future):")
    print("   - Advanced hybrid with reasoning capabilities")
    print("   - Currently maps to hybrid mode")
    
    print(f"\n📋 **CURRENT DEFAULT MODE:** {config.rag_mode}")

def main():
    """Main configuration function"""
    
    print("⚙️  **HYBRID RAG CONFIGURATION TOOL**")
    print("=" * 50)
    
    # Show available modes
    show_prompt_templates()
    
    # Test configuration
    pipeline = test_hybrid_configuration()
    
    if pipeline:
        print("\n🎉 **CONFIGURATION COMPLETE!**")
        print("=" * 40)
        print("💡 **How to use:**")
        print("   1. Your RAG system is now in hybrid mode by default")
        print("   2. Queries will combine document + general knowledge")
        print("   3. You can override mode per query with 'rag_mode' parameter")
        print("   4. Test with: python scripts/test_rag_modes.py")
        
        print("\n📚 **API Usage Examples:**")
        print("   # Use default hybrid mode")
        print("   POST /rag/query")
        print("   {\"question\": \"How does JWT work?\"}")
        print()
        print("   # Force strict mode for specific query")
        print("   POST /rag/query") 
        print("   {\"question\": \"What's in our docs?\", \"rag_mode\": \"strict\"}")
        
    else:
        print("\n❌ **CONFIGURATION FAILED**")
        print("   Please check your environment setup and try again")

if __name__ == "__main__":
    main() 