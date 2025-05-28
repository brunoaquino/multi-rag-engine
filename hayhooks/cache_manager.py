"""
Redis Cache Manager for Haystack Pipeline

This module provides a comprehensive caching layer for embeddings, queries,
and other expensive operations in the Haystack pipeline.
"""

import os
import json
import hashlib
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import redis
from redis.exceptions import ConnectionError, RedisError

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache manager"""
    host: str = "redis"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    
    # TTL settings (in seconds)
    embedding_ttl: int = 86400 * 7  # 7 days
    query_result_ttl: int = 3600    # 1 hour
    document_metadata_ttl: int = 86400 * 30  # 30 days
    
    # Cache prefixes
    embedding_prefix: str = "embedding:"
    query_prefix: str = "query:"
    document_prefix: str = "doc:"
    metadata_prefix: str = "meta:"
    semantic_query_prefix: str = "semantic_query:"  # New prefix for semantic queries
    
    # Performance settings
    max_connections: int = 10
    socket_timeout: float = 5.0
    health_check_interval: int = 30
    
    # Semantic caching settings
    semantic_similarity_threshold: float = 0.85  # Minimum similarity for cache hit
    max_semantic_search_results: int = 50  # Maximum queries to check for similarity
    enable_semantic_caching: bool = True


class CacheManager:
    """
    Redis-based cache manager for Haystack pipeline operations
    
    Features:
    - Document embedding caching
    - Query result caching with TTL
    - Semantic query caching with similarity matching
    - Metadata caching for documents
    - Cache invalidation mechanisms
    - Performance monitoring
    - Fallback graceful degradation
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.redis_client = None
        self.connected = False
        
        # Initialize Redis connection
        self._init_connection()
    
    def _init_connection(self):
        """Initialize Redis connection with fallback"""
        try:
            # Try environment variables first
            redis_host = os.getenv("REDIS_HOST", self.config.host)
            
            # Auto-detect if running outside Docker and adjust host
            if redis_host == "redis":
                # Check if we're inside Docker by looking for .dockerenv
                if not os.path.exists("/.dockerenv"):
                    logger.info("Detected running outside Docker, using localhost instead of redis")
                    redis_host = "localhost"
            
            redis_port = int(os.getenv("REDIS_PORT", str(self.config.port)))
            redis_password = os.getenv("REDIS_PASSWORD", self.config.password)
            redis_db = int(os.getenv("REDIS_DB", str(self.config.db)))
            
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                db=redis_db,
                decode_responses=True,
                socket_timeout=self.config.socket_timeout,
                max_connections=self.config.max_connections
            )
            
            # Test connection
            self.redis_client.ping()
            self.connected = True
            logger.info(f"✅ Connected to Redis at {redis_host}:{redis_port}")
            
        except (ConnectionError, RedisError) as e:
            logger.warning(f"⚠️  Redis connection failed: {e}")
            logger.warning("Cache will operate in fallback mode (no caching)")
            self.connected = False
        except Exception as e:
            logger.error(f"❌ Unexpected error connecting to Redis: {e}")
            self.connected = False
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Convert to numpy arrays for efficient computation
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            similarity = dot_product / (norm_v1 * norm_v2)
            return float(similarity)
        except Exception as e:
            logger.warning(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a consistent cache key from arguments"""
        # Create a string representation of all arguments
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_string = "|".join(key_parts)
        
        # Create a hash for consistent key length
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"{prefix}{key_hash}"
    
    def _serialize_data(self, data: Any) -> str:
        """Serialize data for Redis storage"""
        try:
            return json.dumps(data, default=str, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to serialize data: {e}")
            return None
    
    def _deserialize_data(self, data: str) -> Any:
        """Deserialize data from Redis"""
        try:
            return json.loads(data)
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to deserialize data: {e}")
            return None
    
    def is_connected(self) -> bool:
        """Check if Redis is connected and available"""
        if not self.connected:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except:
            self.connected = False
            return False
    
    def get_embedding_cache(self, text: str, model: str = "text-embedding-ada-002") -> Optional[List[float]]:
        """Get cached embedding for text"""
        if not self.is_connected():
            return None
        
        try:
            cache_key = self._generate_cache_key(
                self.config.embedding_prefix,
                text=text,
                model=model
            )
            
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                result = self._deserialize_data(cached_data)
                logger.debug(f"Cache HIT for embedding: {cache_key[:20]}...")
                return result.get("embedding") if result else None
            
            logger.debug(f"Cache MISS for embedding: {cache_key[:20]}...")
            return None
            
        except Exception as e:
            logger.warning(f"Error getting embedding from cache: {e}")
            return None
    
    def set_embedding_cache(self, text: str, embedding: List[float], model: str = "text-embedding-ada-002") -> bool:
        """Cache embedding for text"""
        if not self.is_connected():
            return False
        
        try:
            cache_key = self._generate_cache_key(
                self.config.embedding_prefix,
                text=text,
                model=model
            )
            
            cache_data = {
                "embedding": embedding,
                "model": model,
                "text_length": len(text),
                "cached_at": datetime.utcnow().isoformat()
            }
            
            serialized_data = self._serialize_data(cache_data)
            if serialized_data:
                self.redis_client.setex(
                    cache_key,
                    self.config.embedding_ttl,
                    serialized_data
                )
                logger.debug(f"Cached embedding: {cache_key[:20]}...")
                return True
            
        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")
        
        return False
    
    def get_query_cache(self, question: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached query result"""
        if not self.is_connected():
            return None
        
        try:
            cache_key = self._generate_cache_key(
                self.config.query_prefix,
                question=question,
                **kwargs
            )
            
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                result = self._deserialize_data(cached_data)
                logger.debug(f"Cache HIT for query: {cache_key[:20]}...")
                return result
            
            logger.debug(f"Cache MISS for query: {cache_key[:20]}...")
            return None
            
        except Exception as e:
            logger.warning(f"Error getting query from cache: {e}")
            return None
    
    def set_query_cache(self, question: str, result: Dict[str, Any], **kwargs) -> bool:
        """Cache query result"""
        if not self.is_connected():
            return False
        
        try:
            cache_key = self._generate_cache_key(
                self.config.query_prefix,
                question=question,
                **kwargs
            )
            
            cache_data = {
                "result": result,
                "question": question,
                "cached_at": datetime.utcnow().isoformat(),
                "query_params": kwargs
            }
            
            serialized_data = self._serialize_data(cache_data)
            if serialized_data:
                self.redis_client.setex(
                    cache_key,
                    self.config.query_result_ttl,
                    serialized_data
                )
                logger.debug(f"Cached query result: {cache_key[:20]}...")
                return True
            
        except Exception as e:
            logger.warning(f"Error caching query result: {e}")
        
        return False
    
    def get_semantic_query_cache(self, question: str, question_embedding: List[float], **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached query result using semantic similarity matching"""
        if not self.is_connected() or not self.config.enable_semantic_caching:
            return None
        
        try:
            # First try exact match
            exact_match = self.get_query_cache(question, **kwargs)
            if exact_match:
                return exact_match
            
            # Search for semantically similar queries
            similar_result = self._find_similar_cached_query(question_embedding, **kwargs)
            
            if similar_result:
                logger.debug(f"Semantic cache HIT for query: {question[:50]}... (similarity: {similar_result['similarity']:.3f})")
                return similar_result['result']
            
            logger.debug(f"Semantic cache MISS for query: {question[:50]}...")
            return None
            
        except Exception as e:
            logger.warning(f"Error getting semantic query result from cache: {e}")
            return None
    
    def set_semantic_query_cache(self, question: str, question_embedding: List[float], result: Dict[str, Any], **kwargs) -> bool:
        """Cache query result with embedding for semantic matching"""
        if not self.is_connected() or not self.config.enable_semantic_caching:
            return False
        
        try:
            # Store regular cache first
            regular_cache_success = self.set_query_cache(question, result, **kwargs)
            
            # Store semantic cache with embedding
            semantic_cache_key = self._generate_cache_key(
                self.config.semantic_query_prefix,
                question=question,
                **kwargs
            )
            
            semantic_cache_data = {
                "result": result,
                "question": question,
                "embedding": question_embedding,
                "cached_at": datetime.utcnow().isoformat(),
                "query_params": kwargs
            }
            
            serialized_data = self._serialize_data(semantic_cache_data)
            if serialized_data:
                self.redis_client.setex(
                    semantic_cache_key,
                    self.config.query_result_ttl,
                    serialized_data
                )
                logger.debug(f"Cached semantic query result: {semantic_cache_key[:20]}...")
                return True
            
        except Exception as e:
            logger.warning(f"Error caching semantic query result: {e}")
        
        return False
    
    def _find_similar_cached_query(self, query_embedding: List[float], **kwargs) -> Optional[Dict[str, Any]]:
        """Find the most similar cached query above threshold"""
        try:
            # Get all semantic query keys with matching parameters
            search_pattern = self._generate_cache_key(
                self.config.semantic_query_prefix,
                "pattern",
                **kwargs
            ).replace("pattern", "*")
            
            # Find all matching semantic query keys
            semantic_keys = self.redis_client.keys(search_pattern)
            
            if not semantic_keys:
                return None
            
            # Limit search to most recent entries for performance
            semantic_keys = semantic_keys[:self.config.max_semantic_search_results]
            
            best_similarity = 0.0
            best_result = None
            
            for key in semantic_keys:
                try:
                    cached_data = self.redis_client.get(key)
                    if not cached_data:
                        continue
                    
                    cached_entry = self._deserialize_data(cached_data)
                    if not cached_entry or 'embedding' not in cached_entry:
                        continue
                    
                    # Calculate similarity
                    similarity = self._cosine_similarity(
                        query_embedding, 
                        cached_entry['embedding']
                    )
                    
                    # Check if above threshold and better than current best
                    if similarity >= self.config.semantic_similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_result = {
                            'result': cached_entry.get('result'),
                            'similarity': similarity,
                            'original_question': cached_entry.get('question')
                        }
                
                except Exception as e:
                    logger.debug(f"Error processing cached entry {key}: {e}")
                    continue
            
            return best_result
            
        except Exception as e:
            logger.warning(f"Error finding similar cached query: {e}")
            return None
    
    def get_document_cache(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get cached document metadata"""
        if not self.is_connected():
            return None
        
        try:
            cache_key = f"{self.config.document_prefix}{doc_id}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                result = self._deserialize_data(cached_data)
                logger.debug(f"Cache HIT for document: {doc_id}")
                return result
            
            logger.debug(f"Cache MISS for document: {doc_id}")
            return None
            
        except Exception as e:
            logger.warning(f"Error getting document from cache: {e}")
            return None
    
    def set_document_cache(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """Cache document metadata"""
        if not self.is_connected():
            return False
        
        try:
            cache_key = f"{self.config.document_prefix}{doc_id}"
            
            cache_data = {
                "metadata": metadata,
                "doc_id": doc_id,
                "cached_at": datetime.utcnow().isoformat()
            }
            
            serialized_data = self._serialize_data(cache_data)
            if serialized_data:
                self.redis_client.setex(
                    cache_key,
                    self.config.document_metadata_ttl,
                    serialized_data
                )
                logger.debug(f"Cached document metadata: {doc_id}")
                return True
            
        except Exception as e:
            logger.warning(f"Error caching document metadata: {e}")
        
        return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        if not self.is_connected():
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries matching pattern: {pattern}")
                return deleted
            return 0
            
        except Exception as e:
            logger.warning(f"Error invalidating cache pattern {pattern}: {e}")
            return 0
    
    def invalidate_document_cache(self, doc_id: str = None) -> int:
        """Invalidate document-related cache entries"""
        if doc_id:
            # Invalidate specific document
            pattern = f"{self.config.document_prefix}{doc_id}"
            return self.invalidate_pattern(pattern)
        else:
            # Invalidate all document caches
            pattern = f"{self.config.document_prefix}*"
            return self.invalidate_pattern(pattern)
    
    def invalidate_embedding_cache(self, text_pattern: str = None) -> int:
        """Invalidate embedding cache entries"""
        if text_pattern:
            # This is complex since embeddings are hashed
            # For now, invalidate all embeddings
            logger.warning("Specific text pattern invalidation not implemented, invalidating all embeddings")
        
        pattern = f"{self.config.embedding_prefix}*"
        return self.invalidate_pattern(pattern)
    
    def invalidate_semantic_query_cache(self, pattern: str = None) -> int:
        """Invalidate semantic query cache entries"""
        if pattern:
            search_pattern = f"{self.config.semantic_query_prefix}*{pattern}*"
        else:
            search_pattern = f"{self.config.semantic_query_prefix}*"
        
        return self.invalidate_pattern(search_pattern)
    
    def clear_all_cache(self) -> bool:
        """Clear all cache entries (use with caution)"""
        if not self.is_connected():
            return False
        
        try:
            # Get all keys
            all_keys = self.redis_client.keys("*")
            if all_keys:
                deleted = self.redis_client.delete(*all_keys)
                logger.warning(f"Cleared ALL cache entries: {deleted} keys deleted")
                return True
            return True
            
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.is_connected():
            return {"connected": False, "error": "Redis not connected"}
        
        try:
            info = self.redis_client.info()
            
            # Count keys by prefix
            prefixes = [
                self.config.embedding_prefix,
                self.config.query_prefix,
                self.config.document_prefix,
                self.config.metadata_prefix,
                self.config.semantic_query_prefix
            ]
            
            key_counts = {}
            for prefix in prefixes:
                keys = self.redis_client.keys(f"{prefix}*")
                key_counts[prefix.rstrip(":")] = len(keys)
            
            return {
                "connected": True,
                "total_keys": info.get("db0", {}).get("keys", 0),
                "memory_usage": info.get("used_memory_human", "unknown"),
                "key_counts": key_counts,
                "uptime": info.get("uptime_in_seconds", 0),
                "cache_hit_rate": info.get("keyspace_hits", 0) / max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0)),
                "semantic_caching": {
                    "enabled": self.config.enable_semantic_caching,
                    "similarity_threshold": self.config.semantic_similarity_threshold,
                    "max_search_results": self.config.max_semantic_search_results
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"connected": False, "error": str(e)}


# Global cache manager instance
_cache_manager = None

def get_cache_manager(config: CacheConfig = None) -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager(config)
    
    return _cache_manager 