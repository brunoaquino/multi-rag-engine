#!/usr/bin/env python3
"""
RAG Pipeline Integration Module

Handles integration with Hayhooks RAG pipeline for embeddings generation,
metadata extraction, and document indexing in Pinecone vector store.
"""

import os
import json
import requests
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    document_id: str
    chunk_id: str
    embedding: List[float]
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0

@dataclass
class IndexingResult:
    """Result of document indexing"""
    document_id: str
    total_chunks: int
    indexed_chunks: int
    failed_chunks: int
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    pinecone_ids: List[str] = None

class RAGPipelineClient:
    """Client for interacting with Hayhooks RAG pipeline"""
    
    def __init__(self, 
                 hayhooks_url: str = "http://hayhooks:1416",
                 pipeline_name: str = "rag_pipeline",
                 use_cache: bool = True):
        """
        Initialize RAG pipeline client
        
        Args:
            hayhooks_url: URL of Hayhooks server
            pipeline_name: Name of the RAG pipeline
            use_cache: Whether to use Redis cache for embeddings
        """
        self.hayhooks_url = hayhooks_url.rstrip('/')
        self.pipeline_name = pipeline_name
        self.use_cache = use_cache
        self.session = requests.Session()
        
        # Set timeout for requests
        self.session.timeout = 30
        
        logger.info(f"Initialized RAG Pipeline Client: {self.hayhooks_url}")
    
    def health_check(self) -> bool:
        """Check if Hayhooks server is healthy"""
        try:
            # Check both the pipeline API (1416) and RAG API (8000)
            pipeline_response = self.session.get(f"{self.hayhooks_url}/status", timeout=5)
            
            rag_url = self.hayhooks_url.replace(":1416", ":8000")
            # For Docker networking, /health endpoint doesn't exist, so we'll check a valid endpoint
            rag_response = self.session.get(f"{rag_url}/docs", timeout=5)
            
            return pipeline_response.status_code == 200 and rag_response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def list_pipelines(self) -> List[str]:
        """List available pipelines"""
        try:
            response = self.session.get(f"{self.hayhooks_url}/")
            if response.status_code == 200:
                data = response.json()
                return list(data.keys()) if isinstance(data, dict) else []
            return []
        except Exception as e:
            logger.error(f"Failed to list pipelines: {e}")
            return []
    
    def generate_embeddings(self, 
                          chunks: List[str], 
                          metadata_list: List[Dict[str, Any]],
                          namespace: str = "default") -> List[EmbeddingResult]:
        """
        Generate embeddings for text chunks
        
        Args:
            chunks: List of text chunks
            metadata_list: List of metadata for each chunk
            namespace: Namespace for the embeddings
            
        Returns:
            List of EmbeddingResult objects
        """
        results = []
        
        for i, (chunk, metadata) in enumerate(zip(chunks, metadata_list)):
            start_time = time.time()
            
            try:
                # Prepare request payload
                payload = {
                    "text": chunk,
                    "metadata": {
                        **metadata,
                        "namespace": namespace,
                        "chunk_index": i,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                # Add cache header if enabled
                headers = {"Content-Type": "application/json"}
                if self.use_cache:
                    headers["X-Use-Cache"] = "true"
                
                # Use the RAG index endpoint (port 8000) instead of pipeline endpoint
                rag_url = self.hayhooks_url.replace(":1416", ":8000")
                
                # Prepare document for indexing
                document = {
                    "content": chunk,
                    "meta": payload["metadata"]
                }
                
                # Make request to RAG index endpoint
                print(f"🔍 DEBUG: Indexing document in namespace: {namespace}")
                payload = {
                    "documents": [document],
                    "namespace": namespace
                }
                print(f"🔍 DEBUG: Payload: {payload}")
                
                response = self.session.post(
                    f"{rag_url}/rag/index",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check if indexing was successful
                    if data.get("success", False):
                        # RAG API doesn't return embeddings, but confirms indexing
                        indexed_count = data.get("indexed_documents", 0)
                        
                        result = EmbeddingResult(
                            document_id=metadata.get("document_id", f"doc_{i}"),
                            chunk_id=metadata.get("chunk_id", f"chunk_{i}"),
                            embedding=[],  # Embeddings are stored in Pinecone, not returned
                            metadata=payload["metadata"],
                            success=True,
                            processing_time=processing_time
                        )
                        
                        logger.info(f"Successfully indexed chunk {i} (indexed: {indexed_count})")
                    else:
                        error_msg = data.get("error", "Unknown indexing error")
                        result = EmbeddingResult(
                            document_id=metadata.get("document_id", f"doc_{i}"),
                            chunk_id=metadata.get("chunk_id", f"chunk_{i}"),
                            embedding=[],
                            metadata=metadata,
                            success=False,
                            error_message=error_msg,
                            processing_time=processing_time
                        )
                        
                        logger.error(f"Failed to index chunk {i}: {error_msg}")
                    
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    result = EmbeddingResult(
                        document_id=metadata.get("document_id", f"doc_{i}"),
                        chunk_id=metadata.get("chunk_id", f"chunk_{i}"),
                        embedding=[],
                        metadata=metadata,
                        success=False,
                        error_message=error_msg,
                        processing_time=processing_time
                    )
                    
                    logger.error(f"Failed to generate embedding for chunk {i}: {error_msg}")
                
                results.append(result)
                
            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = str(e)
                
                result = EmbeddingResult(
                    document_id=metadata.get("document_id", f"doc_{i}"),
                    chunk_id=metadata.get("chunk_id", f"chunk_{i}"),
                    embedding=[],
                    metadata=metadata,
                    success=False,
                    error_message=error_msg,
                    processing_time=processing_time
                )
                
                results.append(result)
                logger.error(f"Exception generating embedding for chunk {i}: {e}")
        
        return results
    
    def index_document(self, 
                      embedding_results: List[EmbeddingResult],
                      namespace: str = "default") -> IndexingResult:
        """
        Index document embeddings in Pinecone
        
        Args:
            embedding_results: List of embedding results
            namespace: Pinecone namespace
            
        Returns:
            IndexingResult object
        """
        start_time = time.time()
        
        try:
            # Prepare vectors for indexing
            vectors = []
            for result in embedding_results:
                if result.success and result.embedding:
                    vector = {
                        "id": f"{result.document_id}_{result.chunk_id}",
                        "values": result.embedding,
                        "metadata": result.metadata
                    }
                    vectors.append(vector)
            
            if not vectors:
                return IndexingResult(
                    document_id=embedding_results[0].document_id if embedding_results else "unknown",
                    total_chunks=len(embedding_results),
                    indexed_chunks=0,
                    failed_chunks=len(embedding_results),
                    success=False,
                    error_message="No valid embeddings to index",
                    processing_time=time.time() - start_time
                )
            
            # Prepare request payload
            payload = {
                "vectors": vectors,
                "namespace": namespace
            }
            
            # Make request to indexing endpoint
            response = self.session.post(
                f"{self.hayhooks_url}/{self.pipeline_name}/index",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract indexing results
                indexed_count = data.get("indexed_count", len(vectors))
                pinecone_ids = data.get("ids", [v["id"] for v in vectors])
                
                result = IndexingResult(
                    document_id=embedding_results[0].document_id,
                    total_chunks=len(embedding_results),
                    indexed_chunks=indexed_count,
                    failed_chunks=len(embedding_results) - indexed_count,
                    success=True,
                    processing_time=processing_time,
                    pinecone_ids=pinecone_ids
                )
                
                logger.info(f"Successfully indexed {indexed_count} chunks in Pinecone")
                
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                result = IndexingResult(
                    document_id=embedding_results[0].document_id,
                    total_chunks=len(embedding_results),
                    indexed_chunks=0,
                    failed_chunks=len(embedding_results),
                    success=False,
                    error_message=error_msg,
                    processing_time=processing_time
                )
                
                logger.error(f"Failed to index document: {error_msg}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            result = IndexingResult(
                document_id=embedding_results[0].document_id if embedding_results else "unknown",
                total_chunks=len(embedding_results),
                indexed_chunks=0,
                failed_chunks=len(embedding_results),
                success=False,
                error_message=error_msg,
                processing_time=processing_time
            )
            
            logger.error(f"Exception indexing document: {e}")
            return result
    
    def process_document_full_pipeline(self, 
                                     chunks: List[str],
                                     metadata_list: List[Dict[str, Any]],
                                     namespace: str = "default") -> Tuple[List[EmbeddingResult], IndexingResult]:
        """
        Process document through full RAG pipeline (embeddings + indexing)
        
        Args:
            chunks: List of text chunks
            metadata_list: List of metadata for each chunk
            namespace: Namespace for processing
            
        Returns:
            Tuple of (embedding_results, indexing_result)
        """
        logger.info(f"Processing document through full RAG pipeline: {len(chunks)} chunks")
        
        # Generate embeddings
        embedding_results = self.generate_embeddings(chunks, metadata_list, namespace)
        
        # Index successful embeddings
        successful_embeddings = [r for r in embedding_results if r.success]
        
        if successful_embeddings:
            indexing_result = self.index_document(successful_embeddings, namespace)
        else:
            indexing_result = IndexingResult(
                document_id=metadata_list[0].get("document_id", "unknown") if metadata_list else "unknown",
                total_chunks=len(chunks),
                indexed_chunks=0,
                failed_chunks=len(chunks),
                success=False,
                error_message="No successful embeddings to index"
            )
        
        return embedding_results, indexing_result
    
    def query_similar_documents(self, 
                              query_text: str,
                              namespace: str = "default",
                              top_k: int = 5,
                              rag_mode: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Query for similar documents using the RAG pipeline
        
        Args:
            query_text: Query text
            namespace: Namespace to search in
            top_k: Number of results to return
            rag_mode: RAG mode to use ('strict', 'hybrid', 'enhanced')
            
        Returns:
            List of similar documents with metadata
        """
        try:
            # Use the RAG query endpoint (port 8000)
            rag_url = self.hayhooks_url.replace(":1416", ":8000")
            
            payload = {
                "question": query_text,
                "top_k": top_k,
                "namespace": namespace,
                "rag_mode": rag_mode
            }
            print(f"🔍 DEBUG: Querying namespace: {namespace}")
            print(f"🔍 DEBUG: Query payload: {payload}")
            
            response = self.session.post(
                f"{rag_url}/rag/query",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success", False):
                    # Format response to match expected structure
                    results = []
                    source_docs = data.get("source_documents", [])
                    for doc in source_docs:
                        result = {
                            "content": doc.get("content", ""),
                            "metadata": doc.get("metadata", {}),
                            "score": doc.get("score", 0.0)
                        }
                        results.append(result)
                    return results
                else:
                    error_msg = data.get("error", "Unknown query error")
                    logger.error(f"Query failed: {error_msg}")
                    return []
            else:
                logger.error(f"Query failed: HTTP {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Exception during query: {e}")
            return []

def create_rag_client(hayhooks_url: str = None, 
                     pipeline_name: str = None,
                     use_cache: bool = True) -> RAGPipelineClient:
    """
    Create RAG pipeline client with default configuration
    
    Args:
        hayhooks_url: URL of Hayhooks server (default: http://hayhooks:1416)
        pipeline_name: Name of the RAG pipeline (default: rag_pipeline)
        use_cache: Whether to use Redis cache
        
    Returns:
        RAGPipelineClient instance
    """
    # Use environment variables or defaults
    hayhooks_url = hayhooks_url or os.getenv("HAYHOOKS_URL") or os.getenv("HAYHOOKS_API_URL", "http://hayhooks:1416")
    pipeline_name = pipeline_name or os.getenv("RAG_PIPELINE_NAME", "rag_pipeline")
    
    return RAGPipelineClient(
        hayhooks_url=hayhooks_url,
        pipeline_name=pipeline_name,
        use_cache=use_cache
    ) 