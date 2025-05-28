#!/usr/bin/env python3
"""
Script de teste local para a API Haystack
"""

import sys
import os
sys.path.insert(0, 'hayhooks')
sys.path.insert(0, 'hayhooks/pipelines')

# Carregar variáveis de ambiente do arquivo local
from dotenv import load_dotenv
load_dotenv('local.env.example')  # Para exemplo, normalmente seria .env

# Importar e testar o pipeline de chat
try:
    from pipelines.chat_pipeline import create_chat_pipeline, run_chat_conversation
    print("✅ Chat pipeline imported successfully")
    
    # Criar pipeline
    pipeline = create_chat_pipeline()
    print("✅ Chat pipeline created successfully")
    
    # Testar uma conversa simples sem API keys (deve falhar mas mostrar a estrutura)
    result = run_chat_conversation(
        pipeline=pipeline,
        user_message="Hello, how are you?",
        session_id="test"
    )
    
    print(f"✅ Response: {result['response']}")
    print(f"✅ Provider: {result['metadata'].get('provider_used', 'unknown')}")
    
except Exception as e:
    print(f"❌ Chat pipeline error: {e}")
    # import traceback
    # traceback.print_exc()

# Testar o pipeline RAG
try:
    from pipelines.rag_pipeline import create_rag_pipeline
    print("\n✅ RAG pipeline imported successfully")
    
    # Criar pipeline RAG (pode falhar se não tiver API keys)
    rag_pipeline = create_rag_pipeline(
        pinecone_index="test-haystack-rag",
        primary_llm="gpt-4o-mini"
    )
    print("✅ RAG pipeline created successfully")
    
except Exception as e:
    print(f"\n⚠️  RAG pipeline error (esperado se não tiver API keys): {e}")

# Testar a API diretamente
try:
    print("\n🚀 Testing direct API...")
    from direct_api import app
    print("✅ Direct API imported successfully")
    
except Exception as e:
    print(f"❌ Direct API error: {e}")
    # import traceback
    # traceback.print_exc()

print("\n🏁 Test completed!") 