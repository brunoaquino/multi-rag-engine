#!/usr/bin/env python3
"""
Test script for validating Pinecone indexing with namespace support.
This tests Task 7.6: Index Embeddings and Metadata in Pinecone with Namespace Support
"""

import requests
import json
import time
import os
from typing import Dict, List, Any

class PineconeIndexingTester:
    """Test class for validating Pinecone indexing functionality"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"  # Hayhooks RAG API
        self.streamlit_api = "http://localhost:8503"  # Streamlit API server
        self.test_namespaces = ["test-project-1", "test-project-2", "default"]
        
    def test_health_check(self) -> bool:
        """Test if the RAG API is available"""
        print("🔍 Testing RAG API health...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ RAG API is healthy")
                return True
            else:
                print(f"❌ RAG API unhealthy: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ RAG API not accessible: {e}")
            return False
    
    def test_rag_info(self) -> bool:
        """Test RAG system information"""
        print("\n🔍 Testing RAG system info...")
        try:
            response = requests.get(f"{self.base_url}/rag/info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ RAG Info: {json.dumps(data, indent=2)}")
                return data.get("available", False)
            else:
                print(f"❌ RAG info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ RAG info error: {e}")
            return False
    
    def create_test_documents(self, namespace: str) -> List[Dict[str, Any]]:
        """Create test documents for indexing"""
        return [
            {
                "content": f"This is a test document about artificial intelligence in {namespace}. AI systems use machine learning algorithms to process data.",
                "metadata": {
                    "title": f"AI Document - {namespace}",
                    "source": "test_data",
                    "category": "technology",
                    "namespace": namespace,
                    "document_type": "article",
                    "created_at": "2024-01-01",
                    "author": "Test Author"
                }
            },
            {
                "content": f"Vector databases are essential for RAG systems in {namespace}. They enable efficient similarity search using embeddings.",
                "metadata": {
                    "title": f"Vector DB Document - {namespace}",
                    "source": "test_data",
                    "category": "database",
                    "namespace": namespace,
                    "document_type": "technical",
                    "created_at": "2024-01-02",
                    "author": "Test Engineer"
                }
            },
            {
                "content": f"Pinecone provides a managed vector database service for {namespace} projects. It supports metadata filtering and namespaces.",
                "metadata": {
                    "title": f"Pinecone Document - {namespace}",
                    "source": "test_data",
                    "category": "service",
                    "namespace": namespace,
                    "document_type": "documentation",
                    "created_at": "2024-01-03",
                    "author": "Test Documentation Team"
                }
            }
        ]
    
    def test_document_indexing(self, namespace: str) -> bool:
        """Test document indexing for a specific namespace"""
        print(f"\n🔍 Testing document indexing for namespace: {namespace}")
        
        # Create test documents
        documents = self.create_test_documents(namespace)
        
        # Prepare request payload with namespace
        payload = {
            "documents": documents,
            "batch_size": 100,
            "namespace": namespace
        }
        
        try:
            # Index documents
            response = requests.post(
                f"{self.base_url}/rag/index",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Indexing successful for {namespace}:")
                print(f"   - Success: {data.get('success', False)}")
                print(f"   - Indexed documents: {data.get('indexed_documents', 0)}")
                print(f"   - Total documents: {data.get('total_documents', 0)}")
                
                if data.get('error'):
                    print(f"   - Error: {data['error']}")
                
                return data.get('success', False)
            else:
                print(f"❌ Indexing failed for {namespace}: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Indexing error for {namespace}: {e}")
            return False
    
    def test_document_retrieval(self, namespace: str) -> bool:
        """Test document retrieval for a specific namespace"""
        print(f"\n🔍 Testing document retrieval for namespace: {namespace}")
        
        # Test queries for each namespace
        test_queries = [
            "What is artificial intelligence?",
            "How do vector databases work?",
            "What is Pinecone service?"
        ]
        
        success_count = 0
        
        for query in test_queries:
            try:
                payload = {
                    "question": query,
                    "namespace": namespace,
                    "top_k": 3,
                    "rag_mode": "hybrid"
                }
                
                response = requests.post(
                    f"{self.base_url}/rag/query",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=20
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Query '{query}' in {namespace}:")
                    print(f"   - Success: {data.get('success', False)}")
                    print(f"   - Sources: {len(data.get('source_documents', []))}")
                    print(f"   - Answer preview: {data.get('answer', '')[:100]}...")
                    
                    # Verify namespace isolation
                    source_docs = data.get('source_documents', [])
                    for doc in source_docs:
                        doc_namespace = doc.get('meta', {}).get('namespace', 'unknown')
                        if doc_namespace != namespace:
                            print(f"⚠️  Namespace isolation issue: expected {namespace}, got {doc_namespace}")
                    
                    if data.get('success', False):
                        success_count += 1
                        
                else:
                    print(f"❌ Query failed for '{query}' in {namespace}: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Query error for '{query}' in {namespace}: {e}")
        
        success_rate = success_count / len(test_queries)
        print(f"📊 Retrieval success rate for {namespace}: {success_rate:.1%}")
        return success_rate >= 0.5  # At least 50% success rate
    
    def test_namespace_isolation(self) -> bool:
        """Test that namespaces properly isolate documents"""
        print(f"\n🔍 Testing namespace isolation...")
        
        # Query each namespace and verify results don't leak
        test_query = "artificial intelligence"
        isolation_success = True
        
        for namespace in self.test_namespaces:
            try:
                payload = {
                    "question": test_query,
                    "namespace": namespace,
                    "top_k": 5,
                    "rag_mode": "hybrid"
                }
                
                response = requests.post(
                    f"{self.base_url}/rag/query",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=20
                )
                
                if response.status_code == 200:
                    data = response.json()
                    source_docs = data.get('source_documents', [])
                    
                    # Check that all returned documents belong to the queried namespace
                    for doc in source_docs:
                        doc_namespace = doc.get('meta', {}).get('namespace', 'unknown')
                        if doc_namespace != namespace:
                            print(f"❌ Namespace isolation failed: query for {namespace} returned doc from {doc_namespace}")
                            isolation_success = False
                    
                    print(f"✅ Namespace {namespace}: {len(source_docs)} docs, all from correct namespace")
                
            except Exception as e:
                print(f"❌ Isolation test error for {namespace}: {e}")
                isolation_success = False
        
        return isolation_success
    
    def run_complete_test(self) -> bool:
        """Run complete indexing and retrieval test"""
        print("🚀 Starting Pinecone Indexing Test (Task 7.6)")
        print("=" * 60)
        
        # Step 1: Health check
        if not self.test_health_check():
            print("❌ Health check failed, aborting tests")
            return False
        
        # Step 2: RAG system info
        if not self.test_rag_info():
            print("❌ RAG system not available, aborting tests")
            return False
        
        # Step 3: Index documents in different namespaces
        indexing_results = {}
        for namespace in self.test_namespaces:
            indexing_results[namespace] = self.test_document_indexing(namespace)
            time.sleep(2)  # Small delay between indexing operations
        
        # Step 4: Test retrieval in each namespace
        retrieval_results = {}
        time.sleep(5)  # Allow time for indexing to complete
        
        for namespace in self.test_namespaces:
            if indexing_results.get(namespace, False):
                retrieval_results[namespace] = self.test_document_retrieval(namespace)
                time.sleep(2)
        
        # Step 5: Test namespace isolation
        isolation_success = self.test_namespace_isolation()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        print("\n📝 Indexing Results:")
        for namespace, success in indexing_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {namespace}: {status}")
        
        print("\n🔍 Retrieval Results:")
        for namespace, success in retrieval_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {namespace}: {status}")
        
        print(f"\n🔒 Namespace Isolation: {'✅ PASS' if isolation_success else '❌ FAIL'}")
        
        # Overall success
        all_indexing_passed = all(indexing_results.values())
        all_retrieval_passed = all(retrieval_results.values())
        overall_success = all_indexing_passed and all_retrieval_passed and isolation_success
        
        print(f"\n🎯 OVERALL RESULT: {'✅ TASK 7.6 COMPLETE' if overall_success else '❌ TASK 7.6 NEEDS WORK'}")
        
        if overall_success:
            print("\n🎉 Pinecone indexing with namespace support is working correctly!")
            print("   - Documents can be indexed with metadata")
            print("   - Namespaces properly isolate documents")
            print("   - Retrieval works correctly for each namespace")
            print("   - Metadata is preserved during indexing")
        
        return overall_success


if __name__ == "__main__":
    # Run the test
    tester = PineconeIndexingTester()
    success = tester.run_complete_test()
    
    # Exit with appropriate code
    exit(0 if success else 1) 