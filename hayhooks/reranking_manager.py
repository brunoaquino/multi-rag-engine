"""
Advanced Re-ranking Manager for RAG Pipeline

This module provides sophisticated re-ranking capabilities to improve
document relevance before LLM generation, using multiple scoring strategies
and machine learning models.
"""

import os
import logging
import time
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import re
from collections import Counter

# Haystack imports
from haystack import Document

# Optional dependencies for advanced re-ranking
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query classification types"""
    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"
    ANALYTICAL = "analytical"
    DEFINITION = "definition"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    UNKNOWN = "unknown"


@dataclass
class ReRankingConfig:
    """Configuration for re-ranking system"""
    # Enable/disable features
    enable_hybrid_scoring: bool = True
    enable_cross_encoder: bool = True
    enable_metadata_boost: bool = True
    enable_query_classification: bool = True
    
    # Scoring weights
    semantic_weight: float = 0.6
    bm25_weight: float = 0.2
    metadata_weight: float = 0.1
    length_weight: float = 0.05
    recency_weight: float = 0.05
    
    # Cross-encoder settings
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    cross_encoder_batch_size: int = 32
    cross_encoder_top_k: int = 20  # Number of docs to re-rank with cross-encoder
    
    # BM25 parameters
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    
    # Length normalization
    optimal_doc_length: int = 500  # Characters
    length_penalty_factor: float = 0.1
    
    # Metadata boost factors
    source_authority_boost: Dict[str, float] = None
    content_type_boost: Dict[str, float] = None
    
    # Recency boost (days)
    recency_decay_days: int = 365
    max_recency_boost: float = 0.2
    
    def __post_init__(self):
        if self.source_authority_boost is None:
            self.source_authority_boost = {
                "official": 1.2,
                "documentation": 1.1,
                "trusted": 1.05,
                "user_generated": 0.9,
                "unknown": 1.0
            }
        
        if self.content_type_boost is None:
            self.content_type_boost = {
                "technical_doc": 1.15,
                "faq": 1.1,
                "tutorial": 1.05,
                "blog": 0.95,
                "forum": 0.9,
                "unknown": 1.0
            }


@dataclass
class DocumentScore:
    """Document with comprehensive scoring"""
    document: Document
    original_score: float
    semantic_score: float
    bm25_score: float
    metadata_score: float
    length_score: float
    recency_score: float
    cross_encoder_score: Optional[float] = None
    final_score: float = 0.0
    rank_position: int = 0


class QueryClassifier:
    """Classify query types for adaptive re-ranking"""
    
    def __init__(self):
        # Patterns for different query types
        self.patterns = {
            QueryType.FACTUAL: [
                r'\b(what|who|where|when|which|how many|how much)\b',
                r'\b(is|are|was|were|does|do|did|has|have|had)\b.*\?',
                r'\b(fact|information|data|statistics)\b'
            ],
            QueryType.DEFINITION: [
                r'\b(what is|what are|define|definition|meaning|explain)\b',
                r'\b(means|stands for|refers to)\b'
            ],
            QueryType.PROCEDURAL: [
                r'\b(how to|how do|how can|steps|process|procedure)\b',
                r'\b(install|configure|setup|create|make|build)\b',
                r'\b(tutorial|guide|instructions)\b'
            ],
            QueryType.ANALYTICAL: [
                r'\b(why|analyze|analysis|compare|contrast|evaluate)\b',
                r'\b(advantages|disadvantages|pros|cons|benefits|drawbacks)\b',
                r'\b(impact|effect|influence|relationship)\b'
            ],
            QueryType.COMPARATIVE: [
                r'\b(vs|versus|compare|comparison|difference|similar|unlike)\b',
                r'\b(better|worse|best|worst|prefer|choose)\b'
            ],
            QueryType.TEMPORAL: [
                r'\b(when|before|after|during|timeline|history|recent|latest)\b',
                r'\b(updated|new|old|previous|current)\b'
            ],
            QueryType.CONVERSATIONAL: [
                r'\b(help|please|thanks|thank you|hi|hello)\b',
                r'\b(can you|could you|would you|I need|I want)\b'
            ]
        }
    
    def classify(self, query: str) -> QueryType:
        """Classify query type based on patterns"""
        query_lower = query.lower()
        
        # Score each type
        type_scores = {}
        for query_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            type_scores[query_type] = score
        
        # Return type with highest score
        if max(type_scores.values()) > 0:
            return max(type_scores.keys(), key=lambda k: type_scores[k])
        
        return QueryType.UNKNOWN


class BM25Scorer:
    """BM25 scoring implementation"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}
        self.idf_cache = {}
        self.avgdl = 0
        self.corpus_size = 0
    
    def fit(self, documents: List[Document]):
        """Fit BM25 on document corpus"""
        corpus = [self._tokenize(doc.content) for doc in documents]
        self.corpus_size = len(corpus)
        
        # Calculate document frequencies
        for doc_tokens in corpus:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        
        # Calculate average document length
        if corpus:
            self.avgdl = sum(len(doc) for doc in corpus) / len(corpus)
        
        # Pre-calculate IDF values
        for token, df in self.doc_freqs.items():
            self.idf_cache[token] = math.log((self.corpus_size - df + 0.5) / (df + 0.5))
    
    def score(self, query: str, document: Document) -> float:
        """Calculate BM25 score for query-document pair"""
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document.content)
        doc_len = len(doc_tokens)
        
        if doc_len == 0:
            return 0.0
        
        # Count term frequencies in document
        doc_term_freqs = Counter(doc_tokens)
        
        score = 0.0
        for token in query_tokens:
            if token in self.idf_cache:
                tf = doc_term_freqs.get(token, 0)
                idf = self.idf_cache[token]
                
                # BM25 formula
                numerator = idf * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += numerator / denominator
        
        return score
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens


class HybridScorer:
    """Combine multiple scoring strategies"""
    
    def __init__(self, config: ReRankingConfig):
        self.config = config
        self.bm25_scorer = BM25Scorer(config.bm25_k1, config.bm25_b)
        self.query_classifier = QueryClassifier()
    
    def fit(self, documents: List[Document]):
        """Fit scoring models on document corpus"""
        if self.config.bm25_weight > 0:
            self.bm25_scorer.fit(documents)
    
    def score_documents(self, query: str, documents: List[Document]) -> List[DocumentScore]:
        """Score all documents with hybrid approach"""
        if not documents:
            return []
        
        # Classify query type for adaptive scoring
        query_type = self.query_classifier.classify(query) if self.config.enable_query_classification else QueryType.UNKNOWN
        
        scored_docs = []
        for doc in documents:
            doc_score = DocumentScore(
                document=doc,
                original_score=getattr(doc, 'score', 0.0),
                semantic_score=getattr(doc, 'score', 0.0),
                bm25_score=0.0,
                metadata_score=1.0,
                length_score=1.0,
                recency_score=1.0
            )
            
            # Calculate BM25 score
            if self.config.bm25_weight > 0:
                doc_score.bm25_score = self.bm25_scorer.score(query, doc)
            
            # Calculate metadata score
            if self.config.enable_metadata_boost:
                doc_score.metadata_score = self._calculate_metadata_score(doc)
            
            # Calculate length score
            doc_score.length_score = self._calculate_length_score(doc)
            
            # Calculate recency score
            doc_score.recency_score = self._calculate_recency_score(doc)
            
            # Combine scores with adaptive weights based on query type
            weights = self._get_adaptive_weights(query_type)
            doc_score.final_score = (
                weights['semantic'] * doc_score.semantic_score +
                weights['bm25'] * doc_score.bm25_score +
                weights['metadata'] * doc_score.metadata_score +
                weights['length'] * doc_score.length_score +
                weights['recency'] * doc_score.recency_score
            )
            
            scored_docs.append(doc_score)
        
        return scored_docs
    
    def _calculate_metadata_score(self, document: Document) -> float:
        """Calculate metadata-based boost score"""
        if not document.meta:
            return 1.0
        
        score = 1.0
        
        # Source authority boost
        source_type = document.meta.get('source_type', 'unknown')
        score *= self.config.source_authority_boost.get(source_type, 1.0)
        
        # Content type boost
        content_type = document.meta.get('content_type', 'unknown')
        score *= self.config.content_type_boost.get(content_type, 1.0)
        
        return score
    
    def _calculate_length_score(self, document: Document) -> float:
        """Calculate length-based score adjustment"""
        doc_length = len(document.content)
        optimal_length = self.config.optimal_doc_length
        
        if doc_length == 0:
            return 0.0
        
        # Calculate deviation from optimal length
        length_ratio = doc_length / optimal_length
        if length_ratio > 1:
            # Too long - penalize
            penalty = math.log(length_ratio) * self.config.length_penalty_factor
        else:
            # Too short - smaller penalty
            penalty = (1 - length_ratio) * self.config.length_penalty_factor * 0.5
        
        return max(0.1, 1.0 - penalty)
    
    def _calculate_recency_score(self, document: Document) -> float:
        """Calculate recency-based boost"""
        if not document.meta or 'created_at' not in document.meta:
            return 1.0
        
        try:
            # Assume created_at is a timestamp or parseable date
            created_at = document.meta['created_at']
            if isinstance(created_at, str):
                # Try to parse common date formats
                import datetime
                try:
                    created_date = datetime.datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except:
                    return 1.0
            else:
                created_date = created_at
            
            # Calculate days since creation
            days_old = (datetime.datetime.now(datetime.timezone.utc) - created_date).days
            
            # Apply exponential decay
            decay_factor = math.exp(-days_old / self.config.recency_decay_days)
            boost = self.config.max_recency_boost * decay_factor
            
            return 1.0 + boost
            
        except Exception:
            return 1.0
    
    def _get_adaptive_weights(self, query_type: QueryType) -> Dict[str, float]:
        """Get adaptive weights based on query type"""
        base_weights = {
            'semantic': self.config.semantic_weight,
            'bm25': self.config.bm25_weight,
            'metadata': self.config.metadata_weight,
            'length': self.config.length_weight,
            'recency': self.config.recency_weight
        }
        
        # Adjust weights based on query type
        if query_type == QueryType.FACTUAL:
            # Factual queries benefit from exact term matching
            base_weights['bm25'] *= 1.3
            base_weights['semantic'] *= 0.9
        elif query_type == QueryType.PROCEDURAL:
            # Procedural queries benefit from structured content
            base_weights['metadata'] *= 1.2
            base_weights['length'] *= 1.1
        elif query_type == QueryType.TEMPORAL:
            # Temporal queries benefit from recency
            base_weights['recency'] *= 2.0
        elif query_type == QueryType.DEFINITION:
            # Definition queries benefit from semantic similarity
            base_weights['semantic'] *= 1.2
            base_weights['bm25'] *= 0.8
        
        # Normalize weights
        total_weight = sum(base_weights.values())
        return {k: v / total_weight for k, v in base_weights.items()}


class CrossEncoderReRanker:
    """Cross-encoder based re-ranking"""
    
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        
        if CROSS_ENCODER_AVAILABLE:
            try:
                self.model = CrossEncoder(model_name)
                logger.info(f"Loaded cross-encoder model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder model {model_name}: {e}")
        else:
            logger.warning("Cross-encoder not available - install sentence-transformers")
    
    def rerank(self, query: str, scored_docs: List[DocumentScore], top_k: int) -> List[DocumentScore]:
        """Re-rank top documents using cross-encoder"""
        if not self.model or len(scored_docs) <= 1:
            return scored_docs
        
        # Take top_k documents for cross-encoder re-ranking
        top_docs = sorted(scored_docs, key=lambda x: x.final_score, reverse=True)[:top_k]
        
        if len(top_docs) <= 1:
            return scored_docs
        
        try:
            # Prepare query-document pairs
            query_doc_pairs = [(query, doc.document.content) for doc in top_docs]
            
            # Get cross-encoder scores
            cross_scores = self.model.predict(query_doc_pairs)
            
            # Update document scores
            for doc, cross_score in zip(top_docs, cross_scores):
                doc.cross_encoder_score = float(cross_score)
                # Combine with original score (weighted average)
                doc.final_score = 0.7 * float(cross_score) + 0.3 * doc.final_score
            
            # Update remaining documents that weren't re-ranked
            remaining_docs = scored_docs[len(top_docs):]
            
            # Return re-ranked top docs + remaining docs
            reranked_docs = sorted(top_docs, key=lambda x: x.final_score, reverse=True)
            return reranked_docs + remaining_docs
            
        except Exception as e:
            logger.error(f"Cross-encoder re-ranking failed: {e}")
            return scored_docs


class ReRankingManager:
    """Main re-ranking manager"""
    
    def __init__(self, config: ReRankingConfig = None):
        self.config = config or ReRankingConfig()
        self.hybrid_scorer = HybridScorer(self.config)
        self.cross_encoder_reranker = None
        
        if self.config.enable_cross_encoder:
            self.cross_encoder_reranker = CrossEncoderReRanker(
                self.config.cross_encoder_model,
                self.config.cross_encoder_batch_size
            )
        
        # Metrics
        self.reranking_stats = {
            'total_requests': 0,
            'total_documents': 0,
            'avg_processing_time': 0.0,
            'cross_encoder_usage': 0
        }
    
    def fit_corpus(self, documents: List[Document]):
        """Fit re-ranking models on document corpus"""
        logger.info(f"Fitting re-ranking models on {len(documents)} documents...")
        self.hybrid_scorer.fit(documents)
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Re-rank documents using advanced scoring strategies
        
        Args:
            query: User query
            documents: Retrieved documents to re-rank
            top_k: Number of top documents to return
            
        Returns:
            Re-ranked list of documents
        """
        start_time = time.time()
        
        if not documents:
            return documents
        
        try:
            # Stage 1: Hybrid scoring
            scored_docs = self.hybrid_scorer.score_documents(query, documents)
            
            # Stage 2: Cross-encoder re-ranking (if enabled)
            if (self.config.enable_cross_encoder and 
                self.cross_encoder_reranker and 
                len(scored_docs) > 1):
                
                scored_docs = self.cross_encoder_reranker.rerank(
                    query, 
                    scored_docs, 
                    self.config.cross_encoder_top_k
                )
                self.reranking_stats['cross_encoder_usage'] += 1
            
            # Stage 3: Final ranking and selection
            final_ranked = sorted(scored_docs, key=lambda x: x.final_score, reverse=True)
            
            # Add rank position to documents
            for i, doc_score in enumerate(final_ranked):
                doc_score.rank_position = i + 1
                # Update document score attribute
                doc_score.document.score = doc_score.final_score
            
            # Select top_k if specified
            if top_k:
                final_ranked = final_ranked[:top_k]
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(len(documents), processing_time)
            
            logger.debug(f"Re-ranked {len(documents)} documents in {processing_time:.3f}s")
            
            return [doc_score.document for doc_score in final_ranked]
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return documents
    
    def get_document_scores(self, query: str, documents: List[Document]) -> List[DocumentScore]:
        """Get detailed scoring information for analysis"""
        if not documents:
            return []
        
        scored_docs = self.hybrid_scorer.score_documents(query, documents)
        
        if (self.config.enable_cross_encoder and 
            self.cross_encoder_reranker and 
            len(scored_docs) > 1):
            scored_docs = self.cross_encoder_reranker.rerank(
                query, 
                scored_docs, 
                self.config.cross_encoder_top_k
            )
        
        return sorted(scored_docs, key=lambda x: x.final_score, reverse=True)
    
    def _update_stats(self, doc_count: int, processing_time: float):
        """Update performance statistics"""
        self.reranking_stats['total_requests'] += 1
        self.reranking_stats['total_documents'] += doc_count
        
        # Update running average of processing time
        current_avg = self.reranking_stats['avg_processing_time']
        total_requests = self.reranking_stats['total_requests']
        self.reranking_stats['avg_processing_time'] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get re-ranking performance statistics"""
        return {
            **self.reranking_stats,
            'cross_encoder_available': self.cross_encoder_reranker is not None,
            'hybrid_scoring_enabled': self.config.enable_hybrid_scoring,
            'metadata_boost_enabled': self.config.enable_metadata_boost
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.reranking_stats = {
            'total_requests': 0,
            'total_documents': 0,
            'avg_processing_time': 0.0,
            'cross_encoder_usage': 0
        }


# Factory function for easy instantiation
def create_reranking_manager(
    enable_cross_encoder: bool = True,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    **config_kwargs
) -> ReRankingManager:
    """
    Factory function to create a re-ranking manager
    
    Args:
        enable_cross_encoder: Whether to enable cross-encoder re-ranking
        cross_encoder_model: Model name for cross-encoder
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured ReRankingManager instance
    """
    config = ReRankingConfig(
        enable_cross_encoder=enable_cross_encoder,
        cross_encoder_model=cross_encoder_model,
        **config_kwargs
    )
    
    return ReRankingManager(config)


# Global re-ranking manager instance
_reranking_manager: Optional[ReRankingManager] = None


def get_reranking_manager(config: ReRankingConfig = None) -> ReRankingManager:
    """Get the global re-ranking manager (singleton)"""
    global _reranking_manager
    
    if _reranking_manager is None:
        _reranking_manager = ReRankingManager(config)
    return _reranking_manager


def reset_reranking_manager():
    """Reset the global re-ranking manager (for testing)"""
    global _reranking_manager
    _reranking_manager = None 