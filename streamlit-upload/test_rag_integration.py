#!/usr/bin/env python3
"""
Test RAG Integration

Test script to validate RAG pipeline integration including embeddings generation,
Pinecone indexing, and semantic search functionality.
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from rag_integration import RAGPipelineClient, create_rag_client, EmbeddingResult, IndexingResult
    from document_processor import create_document_processor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the streamlit-upload directory")
    sys.exit(1)

def test_rag_client_creation():
    """Test RAG client creation and configuration"""
    print("🔧 Testing RAG Client Creation...")
    
    try:
        # Test with default settings
        client = create_rag_client()
        assert client is not None
        assert client.hayhooks_url == "http://localhost:1416"
        assert client.pipeline_name == "rag_pipeline"
        assert client.use_cache == True
        print("✅ Default RAG client created successfully")
        
        # Test with custom settings
        custom_client = create_rag_client(
            hayhooks_url="http://localhost:8080",
            pipeline_name="custom_pipeline",
            use_cache=False
        )
        assert custom_client.hayhooks_url == "http://localhost:8080"
        assert custom_client.pipeline_name == "custom_pipeline"
        assert custom_client.use_cache == False
        print("✅ Custom RAG client created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG client creation failed: {e}")
        return False

def test_rag_client_methods():
    """Test RAG client methods (without actual server)"""
    print("🔧 Testing RAG Client Methods...")
    
    try:
        client = create_rag_client()
        
        # Test method existence
        assert hasattr(client, 'health_check')
        assert hasattr(client, 'list_pipelines')
        assert hasattr(client, 'generate_embeddings')
        assert hasattr(client, 'index_document')
        assert hasattr(client, 'process_document_full_pipeline')
        assert hasattr(client, 'query_similar_documents')
        print("✅ All required methods exist")
        
        # Test health check (will fail without server, but method should work)
        health_result = client.health_check()
        assert isinstance(health_result, bool)
        print("✅ Health check method works (returned False as expected without server)")
        
        # Test list pipelines (will return empty without server)
        pipelines = client.list_pipelines()
        assert isinstance(pipelines, list)
        print("✅ List pipelines method works")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG client method test failed: {e}")
        return False

def test_embedding_result_dataclass():
    """Test EmbeddingResult dataclass"""
    print("🔧 Testing EmbeddingResult Dataclass...")
    
    try:
        # Test creation
        result = EmbeddingResult(
            document_id="test_doc",
            chunk_id="chunk_0",
            embedding=[0.1, 0.2, 0.3],
            metadata={"test": "data"},
            success=True,
            processing_time=1.5
        )
        
        assert result.document_id == "test_doc"
        assert result.chunk_id == "chunk_0"
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.metadata == {"test": "data"}
        assert result.success == True
        assert result.processing_time == 1.5
        assert result.error_message is None
        print("✅ EmbeddingResult creation successful")
        
        # Test failed result
        failed_result = EmbeddingResult(
            document_id="test_doc",
            chunk_id="chunk_0",
            embedding=[],
            metadata={},
            success=False,
            error_message="Test error"
        )
        
        assert failed_result.success == False
        assert failed_result.error_message == "Test error"
        print("✅ Failed EmbeddingResult creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ EmbeddingResult test failed: {e}")
        return False

def test_indexing_result_dataclass():
    """Test IndexingResult dataclass"""
    print("🔧 Testing IndexingResult Dataclass...")
    
    try:
        # Test successful indexing
        result = IndexingResult(
            document_id="test_doc",
            total_chunks=5,
            indexed_chunks=5,
            failed_chunks=0,
            success=True,
            processing_time=2.0,
            pinecone_ids=["id1", "id2", "id3", "id4", "id5"]
        )
        
        assert result.document_id == "test_doc"
        assert result.total_chunks == 5
        assert result.indexed_chunks == 5
        assert result.failed_chunks == 0
        assert result.success == True
        assert result.processing_time == 2.0
        assert len(result.pinecone_ids) == 5
        print("✅ Successful IndexingResult creation successful")
        
        # Test failed indexing
        failed_result = IndexingResult(
            document_id="test_doc",
            total_chunks=5,
            indexed_chunks=0,
            failed_chunks=5,
            success=False,
            error_message="Indexing failed"
        )
        
        assert failed_result.success == False
        assert failed_result.error_message == "Indexing failed"
        print("✅ Failed IndexingResult creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ IndexingResult test failed: {e}")
        return False

def test_document_to_rag_pipeline():
    """Test document processing to RAG pipeline integration"""
    print("🔧 Testing Document to RAG Pipeline Integration...")
    
    try:
        # Create test document
        test_content = """
        This is a test document for RAG integration.
        It contains multiple sentences and paragraphs.
        
        The document should be processed into chunks.
        Each chunk will be used for embedding generation.
        
        This tests the complete pipeline from document processing
        to RAG integration and potential indexing.
        """
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file_path = f.name
        
        try:
            # Process document
            processor = create_document_processor(chunk_size=50, chunk_overlap=10)
            doc_result = processor.process_document(temp_file_path, namespace="test")
            
            assert doc_result.success
            assert len(doc_result.chunks) > 0
            print(f"✅ Document processed into {len(doc_result.chunks)} chunks")
            
            # Test metadata preparation for RAG
            metadata_list = []
            for idx, chunk in enumerate(doc_result.chunks):
                # Handle both string chunks and DocumentChunk objects
                chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                metadata = {
                    "document_id": doc_result.metadata.filename.replace('.', '_'),
                    "chunk_id": f"chunk_{idx}",
                    "filename": doc_result.metadata.filename,
                    "namespace": "test",
                    "chunk_index": idx,
                    "total_chunks": len(doc_result.chunks),
                    "word_count": len(chunk_text.split()),
                }
                metadata_list.append(metadata)
            
            assert len(metadata_list) == len(doc_result.chunks)
            assert all('document_id' in meta for meta in metadata_list)
            assert all('chunk_id' in meta for meta in metadata_list)
            print(f"✅ Metadata prepared for {len(metadata_list)} chunks")
            
            # Test that RAG client can be created with the data
            client = create_rag_client()
            assert client is not None
            print("✅ RAG client ready for processing")
            
            return True
            
        finally:
            # Clean up
            os.unlink(temp_file_path)
        
    except Exception as e:
        print(f"❌ Document to RAG pipeline test failed: {e}")
        return False

def test_configuration_handling():
    """Test configuration handling for RAG integration"""
    print("🔧 Testing Configuration Handling...")
    
    try:
        # Test environment variable handling
        original_url = os.getenv("HAYHOOKS_URL")
        original_pipeline = os.getenv("RAG_PIPELINE_NAME")
        
        # Set test environment variables
        os.environ["HAYHOOKS_URL"] = "http://test:9999"
        os.environ["RAG_PIPELINE_NAME"] = "test_pipeline"
        
        try:
            client = create_rag_client()
            assert client.hayhooks_url == "http://test:9999"
            assert client.pipeline_name == "test_pipeline"
            print("✅ Environment variables handled correctly")
            
        finally:
            # Restore original environment
            if original_url:
                os.environ["HAYHOOKS_URL"] = original_url
            else:
                os.environ.pop("HAYHOOKS_URL", None)
                
            if original_pipeline:
                os.environ["RAG_PIPELINE_NAME"] = original_pipeline
            else:
                os.environ.pop("RAG_PIPELINE_NAME", None)
        
        # Test override with parameters
        client = create_rag_client(
            hayhooks_url="http://override:8080",
            pipeline_name="override_pipeline"
        )
        assert client.hayhooks_url == "http://override:8080"
        assert client.pipeline_name == "override_pipeline"
        print("✅ Parameter override works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration handling test failed: {e}")
        return False

def test_streamlit_app_integration():
    """Test integration with Streamlit app components"""
    print("🔧 Testing Streamlit App Integration...")
    
    try:
        # Test that app.py imports RAG components
        with open('app.py', 'r') as f:
            app_content = f.read()
        
        # Check imports
        assert 'from rag_integration import' in app_content
        assert 'RAGPipelineClient' in app_content
        assert 'create_rag_client' in app_content
        assert 'EmbeddingResult' in app_content
        assert 'IndexingResult' in app_content
        print("✅ RAG components imported in app.py")
        
        # Check function existence
        assert 'process_documents_with_rag' in app_content
        assert 'render_rag_query_interface' in app_content
        assert 'perform_semantic_search' in app_content
        print("✅ RAG functions exist in app.py")
        
        # Check configuration options
        assert 'generate_embeddings' in app_content
        assert 'index_in_pinecone' in app_content
        assert 'hayhooks_url' in app_content
        assert 'pipeline_name' in app_content
        print("✅ RAG configuration options in app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app integration test failed: {e}")
        return False

def test_error_handling():
    """Test error handling in RAG integration"""
    print("🔧 Testing Error Handling...")
    
    try:
        client = create_rag_client(hayhooks_url="http://nonexistent:9999")
        
        # Test health check with bad URL
        health_result = client.health_check()
        assert health_result == False
        print("✅ Health check fails gracefully with bad URL")
        
        # Test embeddings generation with bad URL (should handle gracefully)
        embedding_results = client.generate_embeddings(
            chunks=["test chunk"],
            metadata_list=[{"test": "metadata"}],
            namespace="test"
        )
        
        assert isinstance(embedding_results, list)
        assert len(embedding_results) == 1
        assert embedding_results[0].success == False
        assert embedding_results[0].error_message is not None
        print("✅ Embedding generation fails gracefully with bad URL")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def run_all_tests():
    """Run all RAG integration tests"""
    print("🚀 Starting RAG Integration Tests...\n")
    
    tests = [
        ("RAG Client Creation", test_rag_client_creation),
        ("RAG Client Methods", test_rag_client_methods),
        ("EmbeddingResult Dataclass", test_embedding_result_dataclass),
        ("IndexingResult Dataclass", test_indexing_result_dataclass),
        ("Document to RAG Pipeline", test_document_to_rag_pipeline),
        ("Configuration Handling", test_configuration_handling),
        ("Streamlit App Integration", test_streamlit_app_integration),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All RAG integration tests passed!")
        print("The RAG pipeline integration is ready for use!")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed. Review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 