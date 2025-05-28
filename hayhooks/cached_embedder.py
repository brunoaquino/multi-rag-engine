"""
Cached Embedder Components for Haystack

This module provides cached versions of Haystack embedders that use Redis
to cache embeddings and avoid repeated API calls.
"""

import logging
from typing import List, Dict, Any, Optional
from haystack import component, Document
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder

from cache_manager import get_cache_manager, CacheManager

logger = logging.getLogger(__name__)


@component
class CachedOpenAITextEmbedder:
    """
    OpenAI Text Embedder with Redis caching
    
    This component wraps the standard OpenAI embedder and adds caching
    to avoid repeated API calls for the same text.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None,
        **kwargs
    ):
        """
        Initialize cached embedder
        
        Args:
            model: OpenAI embedding model to use
            dimensions: Number of dimensions for embeddings
            api_key: OpenAI API key
            api_base_url: Custom API base URL
            cache_manager: Custom cache manager instance
            **kwargs: Additional arguments for OpenAI embedder
        """
        self.model = model
        self.cache_manager = cache_manager or get_cache_manager()
        
        # Initialize the underlying OpenAI embedder
        # Handle API key properly
        if api_key is None:
            from haystack.utils import Secret
            api_key = Secret.from_env_var("OPENAI_API_KEY")
        
        self.embedder = OpenAITextEmbedder(
            model=model,
            dimensions=dimensions,
            api_key=api_key,
            api_base_url=api_base_url,
            **kwargs
        )
        
        logger.info(f"Initialized CachedOpenAITextEmbedder with model: {model}")
    
    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    def run(self, text: str) -> Dict[str, Any]:
        """
        Embed text with caching
        
        Args:
            text: Text to embed
            
        Returns:
            Dictionary with embedding and metadata
        """
        # Try to get from cache first
        cached_embedding = self.cache_manager.get_embedding_cache(text, self.model)
        
        if cached_embedding is not None:
            logger.debug(f"Using cached embedding for text: {text[:50]}...")
            return {
                "embedding": cached_embedding,
                "meta": {
                    "model": self.model,
                    "usage": {"cached": True}
                }
            }
        
        # Cache miss - generate embedding
        logger.debug(f"Generating new embedding for text: {text[:50]}...")
        try:
            result = self.embedder.run(text)
            embedding = result["embedding"]
            
            # Cache the result
            self.cache_manager.set_embedding_cache(text, embedding, self.model)
            
            # Add cache info to metadata
            meta = result.get("meta", {})
            meta["cached"] = False
            
            return {
                "embedding": embedding,
                "meta": meta
            }
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return empty embedding on error
            return {
                "embedding": [],
                "meta": {"error": str(e), "cached": False}
            }


@component
class CachedOpenAIDocumentEmbedder:
    """
    OpenAI Document Embedder with Redis caching
    
    This component wraps the standard OpenAI document embedder and adds caching
    to avoid repeated API calls for the same document content.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None,
        batch_size: int = 32,
        **kwargs
    ):
        """
        Initialize cached document embedder
        
        Args:
            model: OpenAI embedding model to use
            dimensions: Number of dimensions for embeddings
            api_key: OpenAI API key
            api_base_url: Custom API base URL
            cache_manager: Custom cache manager instance
            batch_size: Batch size for processing documents
            **kwargs: Additional arguments for OpenAI embedder
        """
        self.model = model
        self.cache_manager = cache_manager or get_cache_manager()
        self.batch_size = batch_size
        
        # Initialize the underlying OpenAI embedder
        # Handle API key properly
        if api_key is None:
            from haystack.utils import Secret
            api_key = Secret.from_env_var("OPENAI_API_KEY")
        
        self.embedder = OpenAIDocumentEmbedder(
            model=model,
            dimensions=dimensions,
            api_key=api_key,
            api_base_url=api_base_url,
            **kwargs
        )
        
        logger.info(f"Initialized CachedOpenAIDocumentEmbedder with model: {model}")
    
    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Embed documents with caching
        
        Args:
            documents: List of documents to embed
            
        Returns:
            Dictionary with embedded documents and metadata
        """
        embedded_documents = []
        cache_hits = 0
        cache_misses = 0
        documents_to_embed = []
        document_cache_map = {}
        
        # Check cache for each document
        for i, doc in enumerate(documents):
            content = doc.content or ""
            
            # Try to get from cache
            cached_embedding = self.cache_manager.get_embedding_cache(content, self.model)
            
            if cached_embedding is not None:
                # Cache hit - use cached embedding
                doc.embedding = cached_embedding
                embedded_documents.append(doc)
                cache_hits += 1
                logger.debug(f"Cache HIT for document {i}")
            else:
                # Cache miss - add to batch for embedding
                documents_to_embed.append(doc)
                document_cache_map[len(documents_to_embed) - 1] = i
                cache_misses += 1
                logger.debug(f"Cache MISS for document {i}")
        
        # Generate embeddings for documents not in cache
        if documents_to_embed:
            try:
                logger.info(f"Generating embeddings for {len(documents_to_embed)} documents")
                result = self.embedder.run(documents_to_embed)
                new_embedded_docs = result["documents"]
                
                # Cache the new embeddings and add to result
                for j, doc in enumerate(new_embedded_docs):
                    if doc.embedding:
                        content = doc.content or ""
                        self.cache_manager.set_embedding_cache(content, doc.embedding, self.model)
                    embedded_documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error generating document embeddings: {e}")
                # Add documents without embeddings on error
                for doc in documents_to_embed:
                    embedded_documents.append(doc)
        
        # Sort documents back to original order
        # (This is a simplified approach; in practice, you might need more sophisticated ordering)
        
        logger.info(f"Embedding cache stats - Hits: {cache_hits}, Misses: {cache_misses}")
        
        return {
            "documents": embedded_documents,
            "meta": {
                "model": self.model,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "total_documents": len(documents)
            }
        }


class CachedEmbeddingRetriever:
    """
    Cached wrapper for embedding retrievers
    
    This class provides caching for query embeddings in retrieval operations.
    """
    
    def __init__(self, retriever, cache_manager: Optional[CacheManager] = None):
        """
        Initialize cached retriever
        
        Args:
            retriever: The underlying retriever to wrap
            cache_manager: Custom cache manager instance
        """
        self.retriever = retriever
        self.cache_manager = cache_manager or get_cache_manager()
        
        # Try to get embedding model from retriever
        self.embedding_model = getattr(retriever, 'model', 'text-embedding-ada-002')
        
        logger.info(f"Initialized CachedEmbeddingRetriever for model: {self.embedding_model}")
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Retrieve documents with cached query embedding
        
        Args:
            query: Query string
            **kwargs: Additional retriever arguments
            
        Returns:
            Retrieval results
        """
        # Check cache for query embedding
        cached_embedding = self.cache_manager.get_embedding_cache(query, self.embedding_model)
        
        if cached_embedding is not None:
            logger.debug(f"Using cached embedding for query: {query[:50]}...")
            # If retriever supports direct embedding input, use it
            if hasattr(self.retriever, 'run_with_embedding'):
                return self.retriever.run_with_embedding(cached_embedding, **kwargs)
        
        # Standard retrieval (will generate embedding internally)
        logger.debug(f"Generating new embedding for query: {query[:50]}...")
        result = self.retriever.run(query, **kwargs)
        
        # If we can extract the embedding from the result or retriever, cache it
        # This depends on the specific retriever implementation
        # For now, we'll just return the result
        
        return result


def create_cached_embedders(
    model: str = "text-embedding-ada-002",
    dimensions: Optional[int] = None,
    api_key: Optional[str] = None,
    cache_manager: Optional[CacheManager] = None
) -> tuple:
    """
    Create cached text and document embedders
    
    Args:
        model: OpenAI embedding model
        dimensions: Embedding dimensions
        api_key: OpenAI API key
        cache_manager: Custom cache manager
        
    Returns:
        Tuple of (text_embedder, document_embedder)
    """
    text_embedder = CachedOpenAITextEmbedder(
        model=model,
        dimensions=dimensions,
        api_key=api_key,
        cache_manager=cache_manager
    )
    
    document_embedder = CachedOpenAIDocumentEmbedder(
        model=model,
        dimensions=dimensions,
        api_key=api_key,
        cache_manager=cache_manager
    )
    
    return text_embedder, document_embedder 