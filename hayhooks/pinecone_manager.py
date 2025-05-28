"""
Pinecone Connection Manager and Optimization Module

This module provides optimized Pinecone operations with connection pooling,
document store caching, and query optimization for the RAG pipeline.
"""

import os
import logging
import time
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from threading import Lock
import weakref

# Pinecone integration
try:
    from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
    from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PineconeConfig:
    """Optimized Pinecone configuration"""
    index: str = "haystack-rag"
    dimension: int = 1536
    metric: str = "cosine"
    region: str = "us-east-1"
    cloud: str = "aws"
    
    # Connection pooling settings
    max_connections: int = 10
    connection_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Query optimization settings
    default_top_k: int = 5
    max_top_k: int = 20
    similarity_threshold: float = 0.7
    
    # Cache settings
    enable_store_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_cached_stores: int = 50


@dataclass
class QueryMetrics:
    """Metrics for Pinecone query performance"""
    query_count: int = 0
    total_latency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    
    @property
    def average_latency(self) -> float:
        return self.total_latency / max(self.query_count, 1)
    
    @property
    def cache_hit_rate(self) -> float:
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / max(total_requests, 1)


class PineconeConnectionManager:
    """
    Manages Pinecone connections and provides optimized query operations.
    
    Features:
    - Document store caching and reuse
    - Retriever pooling per namespace
    - Connection optimization
    - Performance metrics
    """
    
    def __init__(self, config: PineconeConfig = None):
        self.config = config or PineconeConfig()
        
        # Thread-safe caches
        self._lock = Lock()
        self._document_stores: Dict[str, PineconeDocumentStore] = {}
        self._retrievers: Dict[str, PineconeEmbeddingRetriever] = {}
        self._store_creation_times: Dict[str, float] = {}
        
        # Performance metrics
        self.metrics = QueryMetrics()
        
        # Weak references for cleanup
        self._weak_refs: Dict[str, weakref.ref] = {}
        
        logger.info(f"Initialized Pinecone connection manager with config: {self.config}")
    
    def _get_store_key(self, namespace: Optional[str] = None) -> str:
        """Generate cache key for document store"""
        return f"{self.config.index}:{namespace or 'default'}"
    
    def _is_store_expired(self, store_key: str) -> bool:
        """Check if cached store is expired"""
        if store_key not in self._store_creation_times:
            return True
        
        creation_time = self._store_creation_times[store_key]
        return (time.time() - creation_time) > self.config.cache_ttl
    
    def _cleanup_expired_stores(self):
        """Remove expired stores from cache"""
        current_time = time.time()
        expired_keys = []
        
        for store_key, creation_time in self._store_creation_times.items():
            if (current_time - creation_time) > self.config.cache_ttl:
                expired_keys.append(store_key)
        
        for key in expired_keys:
            self._remove_store_from_cache(key)
        
        logger.debug(f"Cleaned up {len(expired_keys)} expired Pinecone stores")
    
    def _remove_store_from_cache(self, store_key: str):
        """Remove store from all caches"""
        with self._lock:
            self._document_stores.pop(store_key, None)
            self._retrievers.pop(store_key, None)
            self._store_creation_times.pop(store_key, None)
            self._weak_refs.pop(store_key, None)
    
    def _create_document_store(self, namespace: Optional[str] = None) -> PineconeDocumentStore:
        """Create a new Pinecone document store with optimized settings"""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone integration not available")
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        store_config = {
            "index": self.config.index,
            "metric": self.config.metric,
            "dimension": self.config.dimension,
            "spec": {
                "serverless": {
                    "region": self.config.region,
                    "cloud": self.config.cloud
                }
            }
        }
        
        # Add namespace if specified
        if namespace:
            store_config["namespace"] = namespace
        
        try:
            store = PineconeDocumentStore(**store_config)
            logger.info(f"Created new Pinecone document store for namespace: {namespace or 'default'}")
            return store
            
        except Exception as e:
            logger.error(f"Failed to create Pinecone document store: {e}")
            self.metrics.error_count += 1
            raise
    
    def get_document_store(self, namespace: Optional[str] = None, force_new: bool = False) -> PineconeDocumentStore:
        """
        Get or create a document store for the specified namespace.
        
        Args:
            namespace: Pinecone namespace (None for default)
            force_new: Force creation of new store (bypass cache)
            
        Returns:
            PineconeDocumentStore instance
        """
        store_key = self._get_store_key(namespace)
        
        # Check cache first (unless force_new)
        if not force_new and self.config.enable_store_cache:
            with self._lock:
                # Cleanup expired stores periodically
                if len(self._document_stores) > 0 and len(self._document_stores) % 10 == 0:
                    self._cleanup_expired_stores()
                
                # Return cached store if valid
                if store_key in self._document_stores and not self._is_store_expired(store_key):
                    self.metrics.cache_hits += 1
                    logger.debug(f"Retrieved cached document store for: {store_key}")
                    return self._document_stores[store_key]
        
        # Create new store
        self.metrics.cache_misses += 1
        store = self._create_document_store(namespace)
        
        # Cache the store
        if self.config.enable_store_cache:
            with self._lock:
                # Enforce cache size limit
                if len(self._document_stores) >= self.config.max_cached_stores:
                    # Remove oldest entry
                    oldest_key = min(self._store_creation_times.keys(), 
                                   key=lambda k: self._store_creation_times[k])
                    self._remove_store_from_cache(oldest_key)
                
                self._document_stores[store_key] = store
                self._store_creation_times[store_key] = time.time()
                
                # Create weak reference for cleanup
                self._weak_refs[store_key] = weakref.ref(store)
        
        logger.debug(f"Created and cached new document store for: {store_key}")
        return store
    
    def get_retriever(self, namespace: Optional[str] = None, top_k: Optional[int] = None, 
                     force_new: bool = False) -> PineconeEmbeddingRetriever:
        """
        Get or create a retriever for the specified namespace.
        
        Args:
            namespace: Pinecone namespace (None for default)
            top_k: Number of documents to retrieve
            force_new: Force creation of new retriever
            
        Returns:
            PineconeEmbeddingRetriever instance
        """
        effective_top_k = min(top_k or self.config.default_top_k, self.config.max_top_k)
        retriever_key = f"{self._get_store_key(namespace)}:top_k_{effective_top_k}"
        
        # Check cache first
        if not force_new and self.config.enable_store_cache:
            with self._lock:
                if retriever_key in self._retrievers and not self._is_store_expired(retriever_key):
                    self.metrics.cache_hits += 1
                    logger.debug(f"Retrieved cached retriever for: {retriever_key}")
                    return self._retrievers[retriever_key]
        
        # Create new retriever
        self.metrics.cache_misses += 1
        document_store = self.get_document_store(namespace, force_new)
        
        retriever = PineconeEmbeddingRetriever(
            document_store=document_store,
            top_k=effective_top_k
        )
        
        # Cache the retriever
        if self.config.enable_store_cache:
            with self._lock:
                self._retrievers[retriever_key] = retriever
                self._store_creation_times[retriever_key] = time.time()
        
        logger.debug(f"Created and cached new retriever for: {retriever_key}")
        return retriever
    
    def optimize_query_filters(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Optimize metadata filters for better Pinecone performance.
        
        Args:
            filters: Original filters
            
        Returns:
            Optimized filters
        """
        if not filters:
            return None
        
        # Common optimization patterns
        optimized_filters = {}
        
        # Sort filters by selectivity (most selective first)
        filter_selectivity = {
            'namespace': 1,     # Most selective
            'source': 2,
            'category': 3,
            'document_type': 4,
            'author': 5,
            'created_at': 6,    # Least selective for exact matches
        }
        
        # Apply filters in order of selectivity
        sorted_filters = sorted(filters.items(), 
                              key=lambda x: filter_selectivity.get(x[0], 999))
        
        for key, value in sorted_filters:
            if value is not None:
                # Handle different filter types
                if isinstance(value, list):
                    if len(value) == 1:
                        optimized_filters[key] = {"$eq": value[0]}
                    else:
                        optimized_filters[key] = {"$in": value}
                elif isinstance(value, str):
                    optimized_filters[key] = {"$eq": value}
                elif isinstance(value, dict):
                    # Pass through complex filters
                    optimized_filters[key] = value
                else:
                    optimized_filters[key] = {"$eq": value}
        
        logger.debug(f"Optimized filters: {filters} -> {optimized_filters}")
        return optimized_filters if optimized_filters else None
    
    def execute_query(self, query_embedding: List[float], namespace: Optional[str] = None,
                     filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute optimized Pinecone query with metrics tracking.
        
        Args:
            query_embedding: Query vector
            namespace: Pinecone namespace
            filters: Metadata filters
            top_k: Number of results
            
        Returns:
            Query results with documents
        """
        start_time = time.time()
        
        try:
            # Get optimized retriever
            retriever = self.get_retriever(namespace, top_k)
            
            # Optimize filters
            optimized_filters = self.optimize_query_filters(filters)
            
            # Execute query
            result = retriever.run(
                query_embedding=query_embedding,
                filters=optimized_filters
            )
            
            # Update metrics
            query_time = time.time() - start_time
            self.metrics.query_count += 1
            self.metrics.total_latency += query_time
            
            logger.debug(f"Pinecone query completed in {query_time:.3f}s, "
                        f"returned {len(result.get('documents', []))} documents")
            
            return result
            
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Pinecone query failed: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "query_count": self.metrics.query_count,
            "average_latency": self.metrics.average_latency,
            "total_latency": self.metrics.total_latency,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "error_count": self.metrics.error_count,
            "cached_stores": len(self._document_stores),
            "cached_retrievers": len(self._retrievers)
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = QueryMetrics()
        logger.info("Pinecone connection manager metrics reset")
    
    def clear_cache(self):
        """Clear all cached stores and retrievers"""
        with self._lock:
            self._document_stores.clear()
            self._retrievers.clear()
            self._store_creation_times.clear()
            self._weak_refs.clear()
        
        logger.info("Pinecone connection manager cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Pinecone connections"""
        try:
            # Try to get a default document store
            store = self.get_document_store()
            
            # Try to count documents (lightweight operation)
            doc_count = store.count_documents()
            
            return {
                "healthy": True,
                "document_count": doc_count,
                "cached_stores": len(self._document_stores),
                "metrics": self.get_metrics()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "metrics": self.get_metrics()
            }


# Global connection manager instance
_connection_manager: Optional[PineconeConnectionManager] = None
_manager_lock = Lock()


def get_pinecone_manager(config: PineconeConfig = None) -> PineconeConnectionManager:
    """Get the global Pinecone connection manager (singleton)"""
    global _connection_manager
    
    with _manager_lock:
        if _connection_manager is None:
            _connection_manager = PineconeConnectionManager(config)
        return _connection_manager


def reset_pinecone_manager():
    """Reset the global connection manager (for testing)"""
    global _connection_manager
    
    with _manager_lock:
        if _connection_manager:
            _connection_manager.clear_cache()
        _connection_manager = None 