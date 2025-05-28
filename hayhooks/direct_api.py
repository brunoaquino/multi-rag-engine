#!/usr/bin/env python3
# =================================================================
# Direct API for Chat Pipeline
# =================================================================
"""
Direct FastAPI server to serve the chat pipeline.
This runs alongside Hayhooks to provide chat functionality.
"""

import os
import sys
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import json

# Add pipelines to path
sys.path.insert(0, '/app/pipelines')
sys.path.insert(0, 'pipelines')  # For local development
sys.path.insert(0, '.')  # Current directory

# Import our pipelines
try:
    from chat_pipeline import create_chat_pipeline, run_chat_conversation
except ImportError:
    from pipelines.chat_pipeline import create_chat_pipeline, run_chat_conversation

# RAG pipeline import (optional)
try:
    from rag_pipeline import create_rag_pipeline, RAGConfig, Document
    RAG_AVAILABLE = True
except ImportError:
    try:
        from pipelines.rag_pipeline import create_rag_pipeline, RAGConfig, Document
        RAG_AVAILABLE = True
    except ImportError as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"RAG pipeline not available: {e}")
        RAG_AVAILABLE = False

# Cache manager import (optional)
try:
    from cache_manager import get_cache_manager, CacheManager
    CACHE_AVAILABLE = True
except ImportError:
    try:
        from pipelines.cache_manager import get_cache_manager, CacheManager
        CACHE_AVAILABLE = True
    except ImportError as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Cache manager not available: {e}")
        CACHE_AVAILABLE = False

# Visualization endpoints import (optional)
try:
    from visualization_endpoints import router as visualization_router
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Visualization endpoints not available: {e}")
    VISUALIZATION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Haystack Multi-Pipeline API",
    description="Direct API for Haystack pipelines: Chat and RAG with OpenAI/Anthropic/Pinecone integration",
    version="1.0.0"
)

# Global pipeline instances
chat_pipeline = None
rag_pipeline = None
cache_manager = None
memory_data = {}

# Create OpenAI-compatible API router
from fastapi import APIRouter
api_router = APIRouter(prefix="/api")

async def simulate_streaming_response(text: str, chunk_size: int = 3):
    """Simulate streaming by breaking text into chunks."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if i + chunk_size < len(words):
            chunk += " "
        
        # Create OpenAI-compatible streaming chunk
        stream_chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "haystack-chat",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": chunk
                },
                "finish_reason": None
            }]
        }
        
        yield f"data: {json.dumps(stream_chunk)}\n\n"
        await asyncio.sleep(0.1)  # Small delay for realistic streaming
    
    # Send final chunk
    final_chunk = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk", 
        "created": int(time.time()),
        "model": "haystack-chat",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

class ChatMessage(BaseModel):
    """Chat message model."""
    role: str
    content: str

class ChatRequest(BaseModel):
    """Chat request model."""
    messages: List[ChatMessage]
    session_id: Optional[str] = "default"
    model: Optional[str] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    """Chat response model."""
    message: str
    session_id: str
    model_used: str
    success: bool
    error: Optional[str] = None

class RAGQueryRequest(BaseModel):
    """RAG query request model."""
    question: str
    filters: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None
    top_k: Optional[int] = 5
    rag_mode: Optional[str] = "hybrid"  # "strict", "hybrid", or "enhanced"

class RAGIndexRequest(BaseModel):
    """RAG document indexing request model."""
    documents: List[Dict[str, Any]]  # List of documents with content and metadata
    batch_size: Optional[int] = 100
    namespace: Optional[str] = None  # Pinecone namespace for document indexing

class RAGResponse(BaseModel):
    """RAG response model."""
    answer: str
    source_documents: List[Dict[str, Any]]
    provider_used: str
    rag_mode: Optional[str] = None
    success: bool
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the pipelines on startup."""
    global chat_pipeline, rag_pipeline, cache_manager
    
    # Initialize cache manager
    if CACHE_AVAILABLE:
        try:
            logger.info("🚀 Initializing cache manager...")
            cache_manager = get_cache_manager()
            logger.info("✅ Cache manager initialized successfully!")
        except Exception as e:
            logger.warning(f"⚠️  Cache manager initialization failed: {e}")
    else:
        logger.info("⚠️  Cache manager not available")
    
    # Initialize chat pipeline
    try:
        logger.info("🚀 Initializing chat pipeline...")
        chat_pipeline = create_chat_pipeline()
        logger.info("✅ Chat pipeline initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize chat pipeline: {e}")
        raise e
    
    # Initialize RAG pipeline (optional)
    if RAG_AVAILABLE:
        try:
            # Check required environment variables
            required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                logger.warning(f"⚠️  RAG pipeline initialization skipped - missing variables: {missing_vars}")
            else:
                logger.info("🚀 Initializing RAG pipeline...")
                rag_pipeline = create_rag_pipeline(
                    pinecone_index="haystack-rag",
                    primary_llm="gpt-4o-mini"
                )
                logger.info("✅ RAG pipeline initialized successfully!")
        except Exception as e:
            logger.warning(f"⚠️  RAG pipeline initialization failed: {e}")
    else:
        logger.info("⚠️  RAG pipeline not available due to missing dependencies")

@app.get("/")
async def root():
    """Root endpoint."""
    endpoints = {
        "/chat": "POST - Send chat messages",
        "/rag/query": "POST - Query RAG system",
        "/rag/index": "POST - Index documents into RAG",
        "/rag/info": "GET - RAG system information",
        "/health": "GET - Health check",
        "/sessions/{session_id}/clear": "POST - Clear session memory",
        "/cache/stats": "GET - Get cache statistics",
        "/cache/clear": "POST - Clear all cache or by pattern",
        "/cache/clear/embeddings": "POST - Clear embedding cache",
        "/cache/clear/queries": "POST - Clear query cache",
        "/cache/clear/documents": "POST - Clear document cache",
        "/cache/clear/semantic": "POST - Clear semantic cache",
        "/chunking/metrics": "GET - Get chunking performance metrics",
        "/chunking/metrics/reset": "POST - Reset chunking metrics",
        "/pinecone/metrics": "GET - Get Pinecone performance metrics",
        "/pinecone/metrics/reset": "POST - Reset Pinecone metrics",
        "/pinecone/health": "GET - Pinecone connection health check",
        "/pinecone/cache/clear": "POST - Clear Pinecone connection cache",
        "/reranking/metrics": "GET - Get re-ranking performance metrics",
        "/reranking/metrics/reset": "POST - Reset re-ranking metrics",
        "/reranking/config": "GET - Get re-ranking configuration",
        "/reranking/analyze": "POST - Analyze document scoring (for debugging)",
        "/query-processor/metrics": "GET - Get query processor performance metrics",
        "/query-processor/metrics/reset": "POST - Reset query processor metrics",
        "/query-processor/config": "GET - Get query processor configuration",
        "/query-processor/analyze": "POST - Analyze query processing (for debugging)",
        "/load-balancer/metrics": "GET - Get load balancer performance metrics",
        "/load-balancer/metrics/reset": "POST - Reset load balancer metrics",
        "/load-balancer/workers": "GET - Get worker status and statistics",
        "/load-balancer/config": "GET - Get load balancer configuration"
    }
    
    # Add visualization endpoints if available
    if VISUALIZATION_AVAILABLE:
        endpoints.update({
            "/visualize/interface": "GET - Web interface for pipeline visualization",
            "/visualize/health": "GET - Visualization service health check",
            "/visualize/pipelines": "GET - List available pipelines for visualization",
            "/visualize/pipeline/{name}/info": "GET - Get detailed pipeline information",
            "/visualize/pipeline/{name}/visualize": "POST - Generate pipeline visualization",
            "/visualize/pipeline/{name}/mermaid": "GET - Get Mermaid diagram text",
            "/visualize/download/{filename}": "GET - Download generated visualization files",
            "/visualize/mermaid/start-server": "POST - Start local Mermaid server",
            "/visualize/mermaid/stop-server": "POST - Stop local Mermaid server",
            "/visualize/batch/all-pipelines": "GET - Generate visualizations for all pipelines"
        })
    
    return {
        "message": "Haystack Multi-Pipeline API",
        "status": "running",
        "pipelines": {
            "chat": chat_pipeline is not None,
            "rag": rag_pipeline is not None
        },
        "features": {
            "visualization": VISUALIZATION_AVAILABLE,
            "cache": CACHE_AVAILABLE,
            "rag": RAG_AVAILABLE
        },
        "endpoints": endpoints
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipelines": {
            "chat_ready": chat_pipeline is not None,
            "rag_ready": rag_pipeline is not None
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint."""
    if not chat_pipeline:
        raise HTTPException(status_code=500, detail="Chat pipeline not initialized")
    
    start_time = time.time()
    try:
        logger.info(f"Processing chat request for session: {request.session_id}")
        
        # Convert Pydantic messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Prepare conversation history
        conversation_history = []
        for msg in messages[:-1]:  # All except the last message
            conversation_history.append(msg)
        
        # Run the pipeline
        result = run_chat_conversation(
            pipeline=chat_pipeline,
            user_message=user_message,
            conversation_history=conversation_history,
            session_id=request.session_id
        )
        
        response_message = result.get("response", "Sorry, I couldn't generate a response.")
        model_used = result.get("metadata", {}).get("provider_used", "unknown")
        
        logger.info(f"Successfully generated response for session: {request.session_id}")
        
        return ChatResponse(
            message=response_message,
            session_id=request.session_id,
            model_used=model_used,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return ChatResponse(
            message="",
            session_id=request.session_id,
            model_used="error",
            success=False,
            error=str(e)
        )

@app.post("/rag/query", response_model=RAGResponse)
async def rag_query(request: RAGQueryRequest):
    """Query the RAG system."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    start_time = time.time()
    try:
        logger.info(f"Processing RAG query: {request.question[:100]}...")
        
        result = rag_pipeline.query(
            question=request.question,
            filters=request.filters,
            namespace=request.namespace,
            rag_mode=request.rag_mode
        )
        
        # Convert source documents to serializable format
        source_docs = []
        for doc in result.get("source_documents", []):
            source_docs.append({
                "content": doc.content,
                "metadata": doc.meta,
                "score": getattr(doc, 'score', None)
            })
        
        provider_used = result.get("provider_used", "unknown")
        rag_mode_used = result.get("rag_mode", "unknown")
        
        return RAGResponse(
            answer=result.get("answer", "No answer generated"),
            source_documents=source_docs,
            provider_used=provider_used,
            rag_mode=rag_mode_used,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}")
        return RAGResponse(
            answer="",
            source_documents=[],
            provider_used="error",
            rag_mode=request.rag_mode or "unknown",
            success=False,
            error=str(e)
        )

@app.post("/rag/index")
async def rag_index(request: RAGIndexRequest):
    """Index documents into the RAG system."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        namespace_info = f" in namespace '{request.namespace}'" if request.namespace else ""
        logger.info(f"Indexing {len(request.documents)} documents{namespace_info}...")
        
        # Convert to Haystack Document objects with namespace in metadata
        documents = []
        for doc_data in request.documents:
            metadata = doc_data.get("metadata", {})
            
            # Add namespace to metadata if provided
            if request.namespace:
                metadata["namespace"] = request.namespace
            
            documents.append(Document(
                content=doc_data.get("content", ""),
                meta=metadata
            ))
        
        result = rag_pipeline.index_documents(
            documents=documents,
            batch_size=request.batch_size,
            namespace=request.namespace
        )
        

        
        return {
            "success": result.get("success", False),
            "indexed_documents": result.get("indexed_documents", 0),
            "total_documents": result.get("total_documents", 0),
            "namespace": request.namespace,
            "error": result.get("error")
        }
        
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        return {
            "success": False,
            "indexed_documents": 0,
            "total_documents": len(request.documents),
            "namespace": request.namespace,
            "error": str(e)
        }

@app.get("/rag/info")
async def rag_info():
    """Get RAG system information."""
    if not rag_pipeline:
        return {
            "available": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        doc_count = rag_pipeline.get_document_count()
        return {
            "available": True,
            "document_count": doc_count,
            "index_name": rag_pipeline.config.pinecone_index,
            "embedding_model": rag_pipeline.config.embedding_model,
            "primary_llm": rag_pipeline.config.primary_llm
        }
    except Exception as e:
        return {
            "available": True,
            "error": str(e)
        }

@app.post("/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    """Clear session memory."""
    # This would clear the memory for a specific session
    # Implementation depends on how memory is stored in the pipeline
    return {"message": f"Session {session_id} cleared", "success": True}

@app.get("/sessions")
async def list_sessions():
    """List active sessions."""
    # This would return information about active sessions
    return {"sessions": list(memory_data.keys())}

# =================================================================
# Cache Management Endpoints
# =================================================================

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    if not cache_manager:
        return {
            "available": False,
            "error": "Cache manager not initialized"
        }
    
    try:
        stats = cache_manager.get_cache_stats()
        return {
            "available": True,
            "stats": stats
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

@app.post("/cache/clear")
async def clear_cache(pattern: Optional[str] = None):
    """Clear cache entries."""
    if not cache_manager:
        return {
            "success": False,
            "error": "Cache manager not initialized"
        }
    
    try:
        if pattern:
            deleted = cache_manager.invalidate_pattern(pattern)
            return {
                "success": True,
                "message": f"Cleared {deleted} cache entries matching pattern: {pattern}"
            }
        else:
            success = cache_manager.clear_all_cache()
            return {
                "success": success,
                "message": "All cache entries cleared" if success else "Failed to clear cache"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/cache/clear/embeddings")
async def clear_embedding_cache():
    """Clear embedding cache entries."""
    if not cache_manager:
        return {
            "success": False,
            "error": "Cache manager not initialized"
        }
    
    try:
        deleted = cache_manager.invalidate_embedding_cache()
        return {
            "success": True,
            "message": f"Cleared {deleted} embedding cache entries"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/cache/clear/queries")
async def clear_query_cache():
    """Clear query result cache entries."""
    if not cache_manager:
        return {
            "success": False,
            "error": "Cache manager not initialized"
        }
    
    try:
        deleted = cache_manager.invalidate_pattern(cache_manager.config.query_prefix + "*")
        return {
            "success": True,
            "message": f"Cleared {deleted} query cache entries"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/cache/clear/documents")
async def clear_document_cache():
    """Clear document metadata cache entries."""
    if not cache_manager:
        return {
            "success": False,
            "error": "Cache manager not initialized"
        }
    
    try:
        deleted = cache_manager.invalidate_document_cache()
        return {
            "success": True,
            "message": f"Cleared {deleted} document cache entries"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/cache/clear/semantic")
async def clear_semantic_cache():
    """Clear semantic query cache entries."""
    if not cache_manager:
        return {
            "success": False,
            "error": "Cache manager not initialized"
        }
    
    try:
        deleted = cache_manager.invalidate_semantic_query_cache()
        return {
            "success": True,
            "message": f"Cleared {deleted} semantic cache entries"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# =================================================================
# Chunking Performance Endpoints
# =================================================================

@app.get("/chunking/metrics")
async def get_chunking_metrics():
    """Get chunking performance metrics."""
    try:
        # Import chunking metrics if available
        try:
            from pipelines.optimized_chunking import chunking_metrics
        except ImportError:
            from optimized_chunking import chunking_metrics
        
        metrics = chunking_metrics.get_metrics()
        return {
            "available": True,
            "metrics": metrics
        }
    except ImportError:
        return {
            "available": False,
            "error": "Optimized chunking not available"
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

@app.post("/chunking/metrics/reset")
async def reset_chunking_metrics():
    """Reset chunking performance metrics."""
    try:
        # Import chunking metrics if available
        try:
            from pipelines.optimized_chunking import chunking_metrics
        except ImportError:
            from optimized_chunking import chunking_metrics
        
        chunking_metrics.reset_metrics()
        return {
            "success": True,
            "message": "Chunking metrics reset successfully"
        }
    except ImportError:
        return {
            "success": False,
            "error": "Optimized chunking not available"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# =================================================================
# Pinecone Performance Endpoints
# =================================================================

@app.get("/pinecone/metrics")
async def get_pinecone_metrics():
    """Get Pinecone connection manager performance metrics."""
    if not rag_pipeline:
        return {
            "available": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'pinecone_manager') and rag_pipeline.pinecone_manager:
            metrics = rag_pipeline.pinecone_manager.get_metrics()
            return {
                "available": True,
                "metrics": metrics
            }
        else:
            return {
                "available": False,
                "error": "Pinecone connection manager not available"
            }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

@app.post("/pinecone/metrics/reset")
async def reset_pinecone_metrics():
    """Reset Pinecone connection manager metrics."""
    if not rag_pipeline:
        return {
            "success": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'pinecone_manager') and rag_pipeline.pinecone_manager:
            rag_pipeline.pinecone_manager.reset_metrics()
            return {
                "success": True,
                "message": "Pinecone metrics reset successfully"
            }
        else:
            return {
                "success": False,
                "error": "Pinecone connection manager not available"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/pinecone/health")
async def get_pinecone_health():
    """Get Pinecone connection health status."""
    if not rag_pipeline:
        return {
            "healthy": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'pinecone_manager') and rag_pipeline.pinecone_manager:
            health_info = rag_pipeline.pinecone_manager.health_check()
            return health_info
        else:
            return {
                "healthy": False,
                "error": "Pinecone connection manager not available"
            }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e)
        }

@app.post("/pinecone/cache/clear")
async def clear_pinecone_cache():
    """Clear Pinecone connection cache."""
    if not rag_pipeline:
        return {
            "success": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'pinecone_manager') and rag_pipeline.pinecone_manager:
            rag_pipeline.pinecone_manager.clear_cache()
            return {
                "success": True,
                "message": "Pinecone connection cache cleared successfully"
            }
        else:
            return {
                "success": False,
                "error": "Pinecone connection manager not available"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# =================================================================
# Re-ranking Manager Endpoints
# =================================================================

@app.get("/reranking/metrics")
async def get_reranking_metrics():
    """Get re-ranking performance metrics."""
    if not rag_pipeline:
        return {
            "available": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'reranking_manager') and rag_pipeline.reranking_manager:
            metrics = rag_pipeline.reranking_manager.get_stats()
            return {
                "available": True,
                "metrics": metrics
            }
        else:
            return {
                "available": False,
                "error": "Re-ranking manager not available"
            }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

@app.post("/reranking/metrics/reset")
async def reset_reranking_metrics():
    """Reset re-ranking performance metrics."""
    if not rag_pipeline:
        return {
            "success": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'reranking_manager') and rag_pipeline.reranking_manager:
            rag_pipeline.reranking_manager.reset_stats()
            return {
                "success": True,
                "message": "Re-ranking metrics reset successfully"
            }
        else:
            return {
                "success": False,
                "error": "Re-ranking manager not available"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/reranking/config")
async def get_reranking_config():
    """Get current re-ranking configuration."""
    if not rag_pipeline:
        return {
            "available": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'reranking_manager') and rag_pipeline.reranking_manager:
            config = rag_pipeline.reranking_manager.config
            return {
                "available": True,
                "config": {
                    "enable_hybrid_scoring": config.enable_hybrid_scoring,
                    "enable_cross_encoder": config.enable_cross_encoder,
                    "cross_encoder_model": config.cross_encoder_model,
                    "cross_encoder_top_k": config.cross_encoder_top_k,
                    "semantic_weight": config.semantic_weight,
                    "bm25_weight": config.bm25_weight,
                    "metadata_weight": config.metadata_weight,
                    "length_weight": config.length_weight,
                    "recency_weight": config.recency_weight,
                    "bm25_k1": config.bm25_k1,
                    "bm25_b": config.bm25_b,
                    "optimal_doc_length": config.optimal_doc_length
                }
            }
        else:
            return {
                "available": False,
                "error": "Re-ranking manager not available"
            }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

class DocumentAnalysisRequest(BaseModel):
    """Request model for document scoring analysis."""
    query: str
    documents: List[Dict[str, Any]]  # List with content and metadata

@app.post("/reranking/analyze")
async def analyze_document_scoring(request: DocumentAnalysisRequest):
    """Analyze document scoring for debugging purposes."""
    if not rag_pipeline:
        return {
            "success": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'reranking_manager') and rag_pipeline.reranking_manager:
            from haystack import Document
            
            # Convert dict to Document objects
            documents = []
            for doc_data in request.documents:
                doc = Document(
                    content=doc_data.get("content", ""),
                    meta=doc_data.get("meta", {})
                )
                if "score" in doc_data:
                    doc.score = doc_data["score"]
                documents.append(doc)
            
            # Get detailed scoring
            scored_docs = rag_pipeline.reranking_manager.get_document_scores(
                request.query,
                documents
            )
            
            # Format results
            analysis = []
            for scored_doc in scored_docs:
                analysis.append({
                    "content": scored_doc.document.content[:200] + "..." if len(scored_doc.document.content) > 200 else scored_doc.document.content,
                    "original_score": scored_doc.original_score,
                    "semantic_score": scored_doc.semantic_score,
                    "bm25_score": scored_doc.bm25_score,
                    "metadata_score": scored_doc.metadata_score,
                    "length_score": scored_doc.length_score,
                    "recency_score": scored_doc.recency_score,
                    "cross_encoder_score": scored_doc.cross_encoder_score,
                    "final_score": scored_doc.final_score,
                    "rank_position": scored_doc.rank_position
                })
            
            return {
                "success": True,
                "query": request.query,
                "analysis": analysis
            }
        else:
            return {
                "success": False,
                "error": "Re-ranking manager not available"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# =================================================================
# Query Processor Endpoints
# =================================================================

@app.get("/query-processor/metrics")
async def get_query_processor_metrics():
    """Get query processor performance metrics."""
    if not rag_pipeline:
        return {
            "available": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'query_processor') and rag_pipeline.query_processor:
            metrics = rag_pipeline.query_processor.get_stats()
            return {
                "available": True,
                "metrics": metrics
            }
        else:
            return {
                "available": False,
                "error": "Query processor not available"
            }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

@app.post("/query-processor/metrics/reset")
async def reset_query_processor_metrics():
    """Reset query processor performance metrics."""
    if not rag_pipeline:
        return {
            "success": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'query_processor') and rag_pipeline.query_processor:
            rag_pipeline.query_processor.reset_stats()
            return {
                "success": True,
                "message": "Query processor metrics reset successfully"
            }
        else:
            return {
                "success": False,
                "error": "Query processor not available"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/query-processor/config")
async def get_query_processor_config():
    """Get current query processor configuration."""
    if not rag_pipeline:
        return {
            "available": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'query_processor') and rag_pipeline.query_processor:
            config = rag_pipeline.query_processor.config
            return {
                "available": True,
                "config": {
                    "enable_expansion": config.enable_expansion,
                    "enable_entity_extraction": config.enable_entity_extraction,
                    "enable_intent_classification": config.enable_intent_classification,
                    "enable_normalization": config.enable_normalization,
                    "max_expanded_terms": config.max_expanded_terms,
                    "expansion_similarity_threshold": config.expansion_similarity_threshold,
                    "min_entity_confidence": config.min_entity_confidence,
                    "intent_confidence_threshold": config.intent_confidence_threshold,
                    "remove_stopwords": config.remove_stopwords,
                    "lowercase": config.lowercase,
                    "remove_punctuation": config.remove_punctuation
                }
            }
        else:
            return {
                "available": False,
                "error": "Query processor not available"
            }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

class QueryAnalysisRequest(BaseModel):
    """Request model for query processing analysis."""
    query: str

@app.post("/query-processor/analyze")
async def analyze_query_processing(request: QueryAnalysisRequest):
    """Analyze query processing for debugging purposes."""
    if not rag_pipeline:
        return {
            "success": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'query_processor') and rag_pipeline.query_processor:
            processed_query = rag_pipeline.query_processor.process_query(request.query)
            
            return {
                "success": True,
                "analysis": {
                    "original_query": processed_query.original_query,
                    "normalized_query": processed_query.normalized_query,
                    "intent": processed_query.intent.value,
                    "intent_confidence": processed_query.intent_confidence,
                    "entities": [
                        {
                            "text": entity.text,
                            "type": entity.entity_type.value,
                            "start": entity.start,
                            "end": entity.end,
                            "confidence": entity.confidence
                        }
                        for entity in processed_query.entities
                    ],
                    "expanded_terms": processed_query.expanded_terms,
                    "keywords": processed_query.keywords,
                    "processing_time": processed_query.processing_time,
                    "metadata": processed_query.metadata
                }
            }
        else:
            return {
                "success": False,
                "error": "Query processor not available"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# =================================================================
# Load Balancer Endpoints
# =================================================================

@app.get("/load-balancer/metrics")
async def get_load_balancer_metrics():
    """Get load balancer performance metrics."""
    if not rag_pipeline:
        return {
            "available": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'load_balancer') and rag_pipeline.load_balancer:
            metrics = rag_pipeline.load_balancer.get_stats()
            return {
                "available": True,
                "metrics": metrics
            }
        else:
            return {
                "available": False,
                "error": "Load balancer not available"
            }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

@app.post("/load-balancer/metrics/reset")
async def reset_load_balancer_metrics():
    """Reset load balancer performance metrics."""
    if not rag_pipeline:
        return {
            "success": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'load_balancer') and rag_pipeline.load_balancer:
            rag_pipeline.load_balancer.reset_stats()
            return {
                "success": True,
                "message": "Load balancer metrics reset successfully"
            }
        else:
            return {
                "success": False,
                "error": "Load balancer not available"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/load-balancer/workers")
async def get_load_balancer_workers():
    """Get load balancer worker status and statistics."""
    if not rag_pipeline:
        return {
            "available": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'load_balancer') and rag_pipeline.load_balancer:
            stats = rag_pipeline.load_balancer.get_stats()
            return {
                "available": True,
                "workers": stats.get("worker_stats", {}),
                "active_workers": stats.get("active_workers", 0),
                "queue_size": stats.get("current_queue_size", 0)
            }
        else:
            return {
                "available": False,
                "error": "Load balancer not available"
            }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

@app.get("/load-balancer/config")
async def get_load_balancer_config():
    """Get current load balancer configuration."""
    if not rag_pipeline:
        return {
            "available": False,
            "error": "RAG pipeline not initialized"
        }
    
    try:
        if hasattr(rag_pipeline, 'load_balancer') and rag_pipeline.load_balancer:
            config = rag_pipeline.load_balancer.config
            return {
                "available": True,
                "config": {
                    "max_queue_size": config.max_queue_size,
                    "max_workers": config.max_workers,
                    "worker_timeout": config.worker_timeout,
                    "strategy": config.strategy.value,
                    "failure_threshold": config.failure_threshold,
                    "recovery_timeout": config.recovery_timeout,
                    "half_open_max_calls": config.half_open_max_calls,
                    "enable_priority_queue": config.enable_priority_queue,
                    "max_wait_time": config.max_wait_time,
                    "health_check_interval": config.health_check_interval,
                    "health_check_timeout": config.health_check_timeout
                }
            }
        else:
            return {
                "available": False,
                "error": "Load balancer not available"
            }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

# =================================================================
# Model Manager Endpoints
# =================================================================

# Model manager import (optional)
try:
    from model_manager import get_model_manager, ModelManager, ProviderType, Priority
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("Model manager not available")
    MODEL_MANAGER_AVAILABLE = False

@app.get("/model-manager/status")
async def get_model_manager_status():
    """Get model manager status and configuration."""
    if not MODEL_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Model manager not available")
    
    try:
        manager = get_model_manager()
        if not manager:
            return {"status": "not_initialized", "providers": []}
        
        # Get provider status
        providers = []
        for provider_type in ProviderType:
            provider = manager.provider_manager.get_provider(provider_type)
            if provider:
                providers.append({
                    "type": provider_type.value,
                    "status": provider.health_status.value,
                    "models": list(provider.models.keys()),
                    "rate_limit": provider.rate_limiter.capacity,
                    "current_tokens": provider.rate_limiter.tokens
                })
        
        return {
            "status": "initialized",
            "providers": providers,
            "fallback_chains": list(manager.fallback_chains.keys())
        }
    except Exception as e:
        logger.error(f"Error getting model manager status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-manager/metrics")
async def get_model_manager_metrics():
    """Get model manager metrics and statistics."""
    if not MODEL_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Model manager not available")
    
    try:
        manager = get_model_manager()
        if not manager:
            return {"error": "Model manager not initialized"}
        
        # Get provider statistics
        provider_stats = {}
        for provider_type in ProviderType:
            provider = manager.provider_manager.get_provider(provider_type)
            if provider:
                stats = provider.get_statistics()
                provider_stats[provider_type.value] = {
                    "total_requests": stats.total_requests,
                    "successful_requests": stats.successful_requests,
                    "failed_requests": stats.failed_requests,
                    "total_tokens": stats.total_tokens,
                    "total_cost_usd": stats.total_cost_usd,
                    "average_latency_ms": stats.average_latency_ms,
                    "success_rate": stats.success_rate
                }
        
        return {
            "provider_statistics": provider_stats,
            "cost_optimizer_stats": {
                "total_optimizations": manager.cost_optimizer.total_optimizations,
                "total_savings_usd": manager.cost_optimizer.total_savings_usd
            }
        }
    except Exception as e:
        logger.error(f"Error getting model manager metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model-manager/metrics/reset")
async def reset_model_manager_metrics():
    """Reset model manager metrics."""
    if not MODEL_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Model manager not available")
    
    try:
        manager = get_model_manager()
        if not manager:
            return {"error": "Model manager not initialized"}
        
        # Reset provider statistics
        for provider_type in ProviderType:
            provider = manager.provider_manager.get_provider(provider_type)
            if provider:
                provider.reset_statistics()
        
        # Reset cost optimizer stats
        manager.cost_optimizer.total_optimizations = 0
        manager.cost_optimizer.total_savings_usd = 0.0
        
        return {"message": "Model manager metrics reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting model manager metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class GenerationTestRequest(BaseModel):
    """Request model for testing generation."""
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    fallback_chain: Optional[str] = "default"
    priority: Optional[str] = "normal"

@app.post("/model-manager/test")
async def test_model_generation(request: GenerationTestRequest):
    """Test model generation with the model manager."""
    if not MODEL_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Model manager not available")
    
    try:
        manager = get_model_manager()
        if not manager:
            raise HTTPException(status_code=503, detail="Model manager not initialized")
        
        # Map priority string to enum
        priority_mapping = {
            "low": Priority.LOW,
            "normal": Priority.NORMAL,
            "high": Priority.HIGH,
            "critical": Priority.CRITICAL
        }
        priority = priority_mapping.get(request.priority, Priority.NORMAL)
        
        # Create generation request
        from model_manager import GenerationRequest
        gen_request = GenerationRequest(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            fallback_chain=request.fallback_chain,
            priority=priority,
            metadata={"test": True, "timestamp": time.time()}
        )
        
        # Generate response
        response = await manager.generate(gen_request)
        
        return {
            "text": response.text,
            "model_used": response.model_used,
            "provider_used": response.provider_used.value,
            "tokens_used": response.tokens_used,
            "cost_usd": response.cost_usd,
            "latency_ms": response.latency_ms,
            "fallback_used": response.fallback_used,
            "retries_attempted": response.retries_attempted,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error in model generation test: {e}")
        return {
            "error": str(e),
            "success": False
        }

# =================================================================
# OpenAI-Compatible API Endpoints
# =================================================================

class OpenAIModel(BaseModel):
    """OpenAI model representation."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "haystack"

@api_router.get("/models")
async def list_models():
    """List available models (OpenAI-compatible) - Only RAG model for user interface."""
    models = []
    
    # Only add RAG model for user interface
    if rag_pipeline:
        models.append({
            "id": "haystack-rag",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "haystack"
        })
    
    # If RAG is not available, show a fallback chat model
    if not models and chat_pipeline:
        models.append({
            "id": "haystack-chat",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "haystack"
        })
    
    return {"object": "list", "data": models}

@api_router.post("/chat/completions")
async def openai_chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint with streaming support."""
    
    # Route to appropriate pipeline based on model
    if request.model and "rag" in request.model.lower():
        # Use RAG pipeline
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
        # Get the last user message for RAG query
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        try:
            result = rag_pipeline.query(
                question=user_message
            )
            
            response_message = result.get("answer", "No answer generated")
            model_used = result.get("provider_used", "haystack-rag")
            
            # Handle streaming for RAG
            if request.stream:
                return StreamingResponse(
                    simulate_streaming_response(response_message),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_message
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(response_message.split()),
                    "total_tokens": len(user_message.split()) + len(response_message.split())
                }
            }
            
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")
    
    else:
        # Use chat pipeline
        if not chat_pipeline:
            raise HTTPException(status_code=503, detail="Chat pipeline not available")
        
        try:
            # Convert to our internal format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Get the last user message
            user_message = None
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_message = msg["content"]
                    break
            
            if not user_message:
                raise HTTPException(status_code=400, detail="No user message found")
            
            # Prepare conversation history
            conversation_history = []
            for msg in messages[:-1]:  # All except the last message
                conversation_history.append(msg)
            
            # Run the pipeline
            result = run_chat_conversation(
                pipeline=chat_pipeline,
                user_message=user_message,
                conversation_history=conversation_history,
                session_id=request.session_id or "default"
            )
            
            response_message = result.get("response", "Sorry, I couldn't generate a response.")
            model_used = result.get("metadata", {}).get("provider_used", "unknown")
            
            # Handle streaming for chat
            if request.stream:
                return StreamingResponse(
                    simulate_streaming_response(response_message),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_message
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(response_message.split()),
                    "total_tokens": len(user_message.split()) + len(response_message.split())
                }
            }
            
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

# Include the API router after all endpoints are defined
app.include_router(api_router)

# Include visualization router if available
if VISUALIZATION_AVAILABLE:
    app.include_router(visualization_router)
    logger.info("✅ Visualization endpoints loaded successfully!")
else:
    logger.warning("⚠️  Visualization endpoints not available")

def main():
    """Run the direct API server."""
    logger.info("🚀 Starting Direct Chat API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main() 