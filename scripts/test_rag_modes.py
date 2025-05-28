#!/usr/bin/env python3
"""
Test script for different RAG modes in Haystack RAG system

This script demonstrates the differences between:
- strict mode: Only uses document information
- hybrid mode: Combines document information with general knowledge

Run this after your system is up and you have some documents indexed.
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
NAMESPACE = "documents"  # Change to your namespace

def test_rag_query(question: str, rag_mode: str = None) -> Dict[str, Any]:
    """Test a RAG query with specified mode"""
    
    payload = {
        "question": question,
        "namespace": NAMESPACE
    }
    
    if rag_mode:
        payload["rag_mode"] = rag_mode
    
    try:
        response = requests.post(f"{API_BASE_URL}/rag/query", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error querying RAG: {e}")
        return None

def print_response(response: Dict[str, Any], mode: str):
    """Print formatted response"""
    if not response:
        return
        
    print(f"\n🔹 **{mode.upper()} MODE**")
    print(f"   **Answer:** {response.get('answer', 'No answer')}")
    print(f"   **Mode Used:** {response.get('rag_mode', 'unknown')}")
    print(f"   **Provider:** {response.get('provider_used', 'unknown')}")
    print(f"   **Source Docs:** {len(response.get('source_documents', []))}")
    
    # Show first source document if available
    source_docs = response.get('source_documents', [])
    if source_docs:
        first_doc = source_docs[0]
        content_preview = first_doc.get('content', '')[:100] + "..."
        print(f"   **First Source:** {content_preview}")
    
    print("-" * 80)

def test_comparison():
    """Test questions that highlight the difference between modes"""
    
    print("🚀 **HAYSTACK RAG MODES COMPARISON TEST**")
    print("=" * 80)
    
    # Test questions that will show the difference
    test_questions = [
        "O que é Retrieval-Augmented Generation e quais são suas principais vantagens?",
        "Como implementar autenticação JWT em uma aplicação web?", 
        "Quais são as melhores práticas para otimização de performance em aplicações Python?",
        "Como configurar um sistema de cache Redis para alta performance?",
        "Quais são os principais conceitos de machine learning para iniciantes?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 **TESTE {i}: {question}**")
        print("=" * 80)
        
        # Test strict mode
        strict_response = test_rag_query(question, "strict")
        print_response(strict_response, "strict")
        
        # Test hybrid mode  
        hybrid_response = test_rag_query(question, "hybrid")
        print_response(hybrid_response, "hybrid")
        
        # Compare responses
        if strict_response and hybrid_response:
            strict_length = len(strict_response.get('answer', ''))
            hybrid_length = len(hybrid_response.get('answer', ''))
            
            print(f"📊 **COMPARAÇÃO:**")
            print(f"   - Strict: {strict_length} caracteres")
            print(f"   - Hybrid: {hybrid_length} caracteres")
            print(f"   - Diferença: {hybrid_length - strict_length:+d} caracteres")
        
        # Pause between questions
        if i < len(test_questions):
            time.sleep(2)

def test_knowledge_enhancement():
    """Test how hybrid mode enhances responses with general knowledge"""
    
    print("\n🧠 **TESTE DE ENRIQUECIMENTO DE CONHECIMENTO**")
    print("=" * 80)
    
    # Questions designed to test knowledge enhancement
    enhancement_questions = [
        "Como funciona o HTTPS? Explique o processo completo de handshake.",
        "Quais são os principais padrões de design em Python?",
        "Como implementar autenticação OAuth 2.0 passo a passo?",
        "Explique os algoritmos de machine learning mais importantes.",
        "Como otimizar consultas SQL para melhor performance?"
    ]
    
    for question in enhancement_questions:
        print(f"\n❓ **Pergunta:** {question}")
        print("-" * 50)
        
        # Only test hybrid mode for knowledge enhancement
        response = test_rag_query(question, "hybrid")
        
        if response:
            answer = response.get('answer', '')
            
            # Check for knowledge indicators
            knowledge_indicators = [
                "Additionally, from general knowledge",
                "From general knowledge",
                "It's also important to note",
                "Furthermore,",
                "Generally speaking,",
                "In practice,",
                "Common best practices include"
            ]
            
            found_indicators = [ind for ind in knowledge_indicators if ind.lower() in answer.lower()]
            
            print(f"💡 **Resposta:** {answer[:200]}...")
            print(f"🔍 **Indicadores de conhecimento encontrados:** {len(found_indicators)}")
            if found_indicators:
                print(f"   - {', '.join(found_indicators[:3])}")
        
        time.sleep(1)

def check_system_health():
    """Check if the system is ready for testing"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        health_data = response.json()
        
        if not health_data.get("pipelines", {}).get("rag_ready", False):
            print("❌ RAG pipeline is not ready. Please ensure:")
            print("   - System is running (docker-compose up -d)")
            print("   - API keys are configured")
            print("   - Documents are indexed")
            return False
            
        # Check if we have documents
        info_response = requests.get(f"{API_BASE_URL}/rag/info")
        if info_response.status_code == 200:
            info_data = info_response.json()
            doc_count = info_data.get("document_count", 0)
            print(f"✅ System ready! Documents indexed: {doc_count}")
            return doc_count > 0
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ System health check failed: {e}")
        print("   Make sure the system is running: docker-compose up -d")
        return False

def main():
    """Main test function"""
    print("🔬 **HAYSTACK RAG MODES TESTING SCRIPT**")
    print("=" * 50)
    
    # Check system health
    if not check_system_health():
        return
    
    # Run comparison tests
    test_comparison()
    
    # Run knowledge enhancement tests
    test_knowledge_enhancement()
    
    print("\n🎉 **TESTE CONCLUÍDO!**")
    print("=" * 50)
    print("💡 **RESUMO:**")
    print("   - **Strict Mode:** Respostas baseadas apenas nos documentos")
    print("   - **Hybrid Mode:** Combina documentos + conhecimento global")
    print("   - **Recomendação:** Use hybrid para respostas mais completas")

if __name__ == "__main__":
    main() 