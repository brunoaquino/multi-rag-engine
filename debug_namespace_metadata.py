#!/usr/bin/env python3
"""
Debug script to test namespace metadata preservation in Pinecone indexing
"""

import requests
import json
import time

def test_single_document_debug():
    """Test a single document to debug metadata issues"""
    
    base_url = "http://localhost:8000"
    
    # Test document with explicit namespace
    test_doc = {
        "content": "This is a DEBUG test document for namespace testing. The document should contain metadata with namespace information.",
        "metadata": {
            "title": "Debug Document",
            "source": "debug_test",
            "namespace": "debug-namespace",
            "debug_id": "debug-001",
            "test_type": "namespace_debug"
        }
    }
    
    print("🐛 DEBUGGING NAMESPACE METADATA PRESERVATION")
    print("=" * 60)
    
    # 1. Index the document
    print("1️⃣  Indexing debug document...")
    index_payload = {
        "documents": [test_doc],
        "namespace": "debug-namespace",
        "batch_size": 1
    }
    
    print(f"📤 Index request payload:")
    print(json.dumps(index_payload, indent=2))
    
    response = requests.post(
        f"{base_url}/rag/index",
        json=index_payload,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    
    print(f"📥 Index response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    if response.status_code != 200 or not response.json().get("success"):
        print("❌ Indexing failed!")
        return False
    
    # 2. Wait a bit for indexing to complete
    print("\n⏱️  Waiting for indexing to complete...")
    time.sleep(5)
    
    # 3. Query the document back
    print("\n2️⃣  Querying debug document...")
    query_payload = {
        "question": "debug test document namespace",
        "namespace": "debug-namespace",
        "top_k": 5,
        "rag_mode": "hybrid"
    }
    
    print(f"📤 Query request payload:")
    print(json.dumps(query_payload, indent=2))
    
    response = requests.post(
        f"{base_url}/rag/query",
        json=query_payload,
        headers={"Content-Type": "application/json"},
        timeout=20
    )
    
    print(f"📥 Query response: {response.status_code}")
    query_result = response.json()
    
    if response.status_code != 200:
        print("❌ Query failed!")
        return False
    
    # 4. Analyze the results
    print("\n3️⃣  Analyzing retrieved documents...")
    source_docs = query_result.get("source_documents", [])
    
    print(f"📊 Found {len(source_docs)} source documents")
    
    for i, doc in enumerate(source_docs):
        print(f"\n📄 Document {i+1}:")
        print(f"   Content preview: {doc.get('content', '')[:100]}...")
        print(f"   Metadata: {json.dumps(doc.get('metadata', {}), indent=6)}")
        
        # Check if this is our debug document
        metadata = doc.get('metadata', {}) or {}
        if isinstance(metadata, dict):
            debug_id = metadata.get('debug_id')
            namespace_in_meta = metadata.get('namespace')
            
            if debug_id == 'debug-001':
                print(f"   ✅ Found our debug document!")
                print(f"   🏷️  Namespace in metadata: {namespace_in_meta}")
                
                if namespace_in_meta == 'debug-namespace':
                    print(f"   ✅ Namespace preserved correctly!")
                    return True
                else:
                    print(f"   ❌ Namespace NOT preserved! Expected 'debug-namespace', got '{namespace_in_meta}'")
                    return False
    
    print("❌ Debug document not found in results!")
    return False

def test_multiple_namespaces_debug():
    """Test multiple namespaces with debug info"""
    
    base_url = "http://localhost:8000"
    
    # Create test documents for different namespaces
    namespaces = ["ns1", "ns2", "ns3"]
    
    print("\n\n🔍 DEBUGGING MULTIPLE NAMESPACES")
    print("=" * 60)
    
    # Index documents in each namespace
    for ns in namespaces:
        print(f"\n📝 Indexing document in namespace: {ns}")
        
        test_doc = {
            "content": f"This is a test document for namespace {ns}. Content is unique to {ns}.",
            "metadata": {
                "title": f"Document for {ns}",
                "namespace": ns,
                "unique_id": f"doc-{ns}-001"
            }
        }
        
        index_payload = {
            "documents": [test_doc],
            "namespace": ns,
            "batch_size": 1
        }
        
        response = requests.post(
            f"{base_url}/rag/index",
            json=index_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"   Index result: {response.json().get('success', False)}")
    
    # Wait for indexing
    print("\n⏱️  Waiting for all indexing to complete...")
    time.sleep(8)
    
    # Query each namespace and check isolation
    for ns in namespaces:
        print(f"\n🔍 Querying namespace: {ns}")
        
        query_payload = {
            "question": f"document namespace {ns}",
            "namespace": ns,
            "top_k": 3,
            "rag_mode": "hybrid"
        }
        
        response = requests.post(
            f"{base_url}/rag/query",
            json=query_payload,
            headers={"Content-Type": "application/json"},
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json()
            docs = result.get("source_documents", [])
            print(f"   📊 Found {len(docs)} documents")
            
            # Check if all documents belong to the correct namespace
            correct_namespace_count = 0
            for doc in docs:
                doc_ns = doc.get('metadata', {}).get('namespace')
                if doc_ns == ns:
                    correct_namespace_count += 1
                print(f"   📄 Doc namespace: {doc_ns} (expected: {ns})")
            
            if correct_namespace_count == len(docs) and len(docs) > 0:
                print(f"   ✅ All documents correctly isolated in {ns}")
            else:
                print(f"   ❌ Namespace isolation failed for {ns}")
        else:
            print(f"   ❌ Query failed for {ns}")

if __name__ == "__main__":
    print("🚀 Starting Namespace Metadata Debug Session")
    
    # Test 1: Single document debug
    single_doc_success = test_single_document_debug()
    
    # Test 2: Multiple namespace debug
    test_multiple_namespaces_debug()
    
    print("\n\n📊 DEBUG SUMMARY")
    print("=" * 40)
    print(f"Single document namespace preservation: {'✅ PASS' if single_doc_success else '❌ FAIL'}")
    
    if not single_doc_success:
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("1. Check if Pinecone DocumentStore is properly configured with namespace support")
        print("2. Verify that metadata is being preserved during document embedding")
        print("3. Check if namespace-specific document stores are working correctly")
        print("4. Investigate metadata serialization in the pipeline") 