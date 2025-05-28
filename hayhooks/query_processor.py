#!/usr/bin/env python3
"""
Enhanced Query Processor for RAG Pipeline Optimization

This module provides advanced query processing capabilities including:
- Query expansion with synonyms and related terms
- Entity recognition and extraction
- Intent classification for different query types
- Query normalization for better matching
- Performance metrics and monitoring

Author: AI Assistant
Date: 2025-05-27
"""

import re
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import unicodedata

# Optional dependencies with graceful fallback
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Query intent classification."""
    FACTUAL = "factual"           # What is X? Who is Y?
    DEFINITION = "definition"     # Define X, What does Y mean?
    COMPARISON = "comparison"     # Compare X and Y, Difference between
    PROCEDURAL = "procedural"     # How to X? Steps to Y?
    ANALYTICAL = "analytical"     # Why X? Analyze Y?
    TEMPORAL = "temporal"         # When X? History of Y?
    CONVERSATIONAL = "conversational"  # General chat, greetings
    UNKNOWN = "unknown"           # Cannot classify

class EntityType(Enum):
    """Named entity types."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    PRODUCT = "product"
    OTHER = "other"

@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float = 1.0

@dataclass
class ProcessedQuery:
    """Represents a processed query with all enhancements."""
    original_query: str
    normalized_query: str
    expanded_terms: List[str]
    entities: List[Entity]
    intent: QueryIntent
    intent_confidence: float
    keywords: List[str]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QueryProcessorConfig:
    """Configuration for query processor."""
    enable_expansion: bool = True
    enable_entity_extraction: bool = True
    enable_intent_classification: bool = True
    enable_normalization: bool = True
    
    # Expansion settings
    max_expanded_terms: int = 10
    expansion_similarity_threshold: float = 0.7
    
    # Entity extraction settings
    min_entity_confidence: float = 0.5
    
    # Intent classification settings
    intent_confidence_threshold: float = 0.6
    
    # Normalization settings
    remove_stopwords: bool = True
    lowercase: bool = True
    remove_punctuation: bool = True

class QueryExpander:
    """Handles query expansion with synonyms and related terms."""
    
    def __init__(self, config: QueryProcessorConfig):
        self.config = config
        self._synonym_cache = {}
        self._cache_lock = threading.Lock()
        
        # Initialize NLTK if available
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                logger.warning("WordNet not found, downloading...")
                try:
                    nltk.download('wordnet', quiet=True)
                    nltk.download('omw-1.4', quiet=True)
                except Exception as e:
                    logger.error(f"Failed to download WordNet: {e}")
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms."""
        if not self.config.enable_expansion or not NLTK_AVAILABLE:
            return []
        
        try:
            # Tokenize and get synonyms
            tokens = word_tokenize(query.lower())
            expanded_terms = set()
            
            for token in tokens:
                if len(token) > 2:  # Skip short words
                    synonyms = self._get_synonyms(token)
                    expanded_terms.update(synonyms[:3])  # Limit synonyms per word
            
            # Remove original query terms
            query_tokens = set(tokens)
            expanded_terms = expanded_terms - query_tokens
            
            return list(expanded_terms)[:self.config.max_expanded_terms]
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return []
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        with self._cache_lock:
            if word in self._synonym_cache:
                return self._synonym_cache[word]
        
        try:
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word and len(synonym) > 2:
                        synonyms.add(synonym)
            
            synonym_list = list(synonyms)
            
            with self._cache_lock:
                self._synonym_cache[word] = synonym_list
            
            return synonym_list
            
        except Exception as e:
            logger.error(f"Synonym extraction failed for '{word}': {e}")
            return []

class EntityExtractor:
    """Handles named entity recognition and extraction."""
    
    def __init__(self, config: QueryProcessorConfig):
        self.config = config
        self.nlp = None
        
        # Initialize spaCy if available
        if SPACY_AVAILABLE and self.config.enable_entity_extraction:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found. Entity extraction will use regex patterns.")
                self.nlp = None
    
    def extract_entities(self, query: str) -> List[Entity]:
        """Extract named entities from query."""
        if not self.config.enable_entity_extraction:
            return []
        
        entities = []
        
        # Use spaCy if available
        if self.nlp:
            entities.extend(self._extract_with_spacy(query))
        
        # Use regex patterns as fallback
        entities.extend(self._extract_with_regex(query))
        
        # Remove duplicates and filter by confidence
        entities = self._deduplicate_entities(entities)
        entities = [e for e in entities if e.confidence >= self.config.min_entity_confidence]
        
        return entities
    
    def _extract_with_spacy(self, query: str) -> List[Entity]:
        """Extract entities using spaCy NLP."""
        try:
            doc = self.nlp(query)
            entities = []
            
            for ent in doc.ents:
                entity_type = self._map_spacy_label(ent.label_)
                entity = Entity(
                    text=ent.text,
                    entity_type=entity_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.9  # spaCy confidence
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"spaCy entity extraction failed: {e}")
            return []
    
    def _extract_with_regex(self, query: str) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []
        
        # Technology terms
        tech_patterns = [
            r'\b(?:AI|ML|API|REST|GraphQL|SQL|NoSQL|Docker|Kubernetes|React|Vue|Angular|Python|JavaScript|TypeScript|Java|C\+\+|Go|Rust)\b',
            r'\b(?:machine learning|artificial intelligence|deep learning|neural network|blockchain|cloud computing)\b'
        ]
        
        for pattern in tech_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                entity = Entity(
                    text=match.group(),
                    entity_type=EntityType.TECHNOLOGY,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                )
                entities.append(entity)
        
        # Date patterns
        date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b'
        for match in re.finditer(date_pattern, query, re.IGNORECASE):
            entity = Entity(
                text=match.group(),
                entity_type=EntityType.DATE,
                start=match.start(),
                end=match.end(),
                confidence=0.9
            )
            entities.append(entity)
        
        return entities
    
    def _map_spacy_label(self, label: str) -> EntityType:
        """Map spaCy entity labels to our EntityType enum."""
        mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,
            'LOC': EntityType.LOCATION,
            'DATE': EntityType.DATE,
            'TIME': EntityType.DATE,
            'PRODUCT': EntityType.PRODUCT,
            'EVENT': EntityType.CONCEPT,
            'WORK_OF_ART': EntityType.CONCEPT,
            'LAW': EntityType.CONCEPT,
            'LANGUAGE': EntityType.CONCEPT,
        }
        return mapping.get(label, EntityType.OTHER)
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on text and position."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.text.lower(), entity.start, entity.end)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities

class IntentClassifier:
    """Classifies query intent based on patterns and keywords."""
    
    def __init__(self, config: QueryProcessorConfig):
        self.config = config
        self._intent_patterns = self._build_intent_patterns()
    
    def classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify the intent of a query."""
        if not self.config.enable_intent_classification:
            return QueryIntent.UNKNOWN, 0.0
        
        query_lower = query.lower().strip()
        intent_scores = defaultdict(float)
        
        # Pattern-based classification
        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    intent_scores[intent] += 1.0
        
        # Keyword-based classification
        self._classify_by_keywords(query_lower, intent_scores)
        
        # Find best intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(intent_scores[best_intent] / 3.0, 1.0)  # Normalize
            
            if confidence >= self.config.intent_confidence_threshold:
                return best_intent, confidence
        
        return QueryIntent.UNKNOWN, 0.0
    
    def _build_intent_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Build regex patterns for intent classification."""
        return {
            QueryIntent.FACTUAL: [
                r'\bwhat\s+is\b',
                r'\bwho\s+is\b',
                r'\bwhere\s+is\b',
                r'\btell\s+me\s+about\b',
                r'\bfacts?\s+about\b'
            ],
            QueryIntent.DEFINITION: [
                r'\bdefine\b',
                r'\bdefinition\s+of\b',
                r'\bwhat\s+does\s+.+\s+mean\b',
                r'\bmeaning\s+of\b',
                r'\bexplain\b'
            ],
            QueryIntent.COMPARISON: [
                r'\bcompare\b',
                r'\bdifference\s+between\b',
                r'\bvs\b',
                r'\bversus\b',
                r'\bbetter\s+than\b',
                r'\bsimilar\s+to\b'
            ],
            QueryIntent.PROCEDURAL: [
                r'\bhow\s+to\b',
                r'\bsteps\s+to\b',
                r'\bguide\s+to\b',
                r'\btutorial\b',
                r'\binstructions?\b'
            ],
            QueryIntent.ANALYTICAL: [
                r'\bwhy\b',
                r'\banalyze\b',
                r'\banalysis\b',
                r'\breason\s+for\b',
                r'\bcause\s+of\b'
            ],
            QueryIntent.TEMPORAL: [
                r'\bwhen\b',
                r'\bhistory\s+of\b',
                r'\btimeline\b',
                r'\bchronology\b',
                r'\bevolution\s+of\b'
            ],
            QueryIntent.CONVERSATIONAL: [
                r'\bhello\b',
                r'\bhi\b',
                r'\bthanks?\b',
                r'\bthank\s+you\b',
                r'\bbye\b',
                r'\bgoodbye\b'
            ]
        }
    
    def _classify_by_keywords(self, query: str, intent_scores: Dict[QueryIntent, float]):
        """Classify intent based on keywords."""
        keyword_mapping = {
            QueryIntent.FACTUAL: ['information', 'data', 'facts', 'details'],
            QueryIntent.DEFINITION: ['meaning', 'definition', 'concept', 'term'],
            QueryIntent.COMPARISON: ['difference', 'comparison', 'contrast', 'similar'],
            QueryIntent.PROCEDURAL: ['process', 'method', 'procedure', 'approach'],
            QueryIntent.ANALYTICAL: ['analysis', 'reason', 'cause', 'explanation'],
            QueryIntent.TEMPORAL: ['time', 'date', 'period', 'era', 'history']
        }
        
        for intent, keywords in keyword_mapping.items():
            for keyword in keywords:
                if keyword in query:
                    intent_scores[intent] += 0.5

class QueryNormalizer:
    """Handles query normalization and preprocessing."""
    
    def __init__(self, config: QueryProcessorConfig):
        self.config = config
        self._stopwords = set()
        
        # Initialize stopwords if available
        if NLTK_AVAILABLE and self.config.remove_stopwords:
            try:
                self._stopwords = set(stopwords.words('english'))
            except LookupError:
                logger.warning("NLTK stopwords not found, downloading...")
                try:
                    nltk.download('stopwords', quiet=True)
                    self._stopwords = set(stopwords.words('english'))
                except Exception as e:
                    logger.error(f"Failed to download stopwords: {e}")
    
    def normalize_query(self, query: str) -> str:
        """Normalize query text."""
        if not self.config.enable_normalization:
            return query
        
        normalized = query
        
        # Unicode normalization
        normalized = unicodedata.normalize('NFKD', normalized)
        
        # Lowercase
        if self.config.lowercase:
            normalized = normalized.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove punctuation (optional)
        if self.config.remove_punctuation:
            normalized = re.sub(r'[^\w\s]', ' ', normalized)
            normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove stopwords (optional)
        if self.config.remove_stopwords and self._stopwords:
            words = normalized.split()
            words = [word for word in words if word not in self._stopwords]
            normalized = ' '.join(words)
        
        return normalized

@dataclass
class QueryProcessorMetrics:
    """Metrics for query processor performance."""
    total_queries: int = 0
    total_processing_time: float = 0.0
    expansion_count: int = 0
    entity_extraction_count: int = 0
    intent_classification_count: int = 0
    normalization_count: int = 0
    
    # Intent distribution
    intent_distribution: Dict[QueryIntent, int] = field(default_factory=lambda: defaultdict(int))
    
    # Entity distribution
    entity_distribution: Dict[EntityType, int] = field(default_factory=lambda: defaultdict(int))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get formatted statistics."""
        avg_time = self.total_processing_time / max(self.total_queries, 1)
        
        return {
            "total_queries": self.total_queries,
            "average_processing_time": avg_time,
            "expansion_usage": self.expansion_count,
            "entity_extraction_usage": self.entity_extraction_count,
            "intent_classification_usage": self.intent_classification_count,
            "normalization_usage": self.normalization_count,
            "intent_distribution": dict(self.intent_distribution),
            "entity_distribution": dict(self.entity_distribution),
            "components_available": {
                "spacy": SPACY_AVAILABLE,
                "nltk": NLTK_AVAILABLE
            }
        }
    
    def reset(self):
        """Reset all metrics."""
        self.total_queries = 0
        self.total_processing_time = 0.0
        self.expansion_count = 0
        self.entity_extraction_count = 0
        self.intent_classification_count = 0
        self.normalization_count = 0
        self.intent_distribution.clear()
        self.entity_distribution.clear()

class EnhancedQueryProcessor:
    """Main query processor that orchestrates all enhancements."""
    
    def __init__(self, config: Optional[QueryProcessorConfig] = None):
        self.config = config or QueryProcessorConfig()
        
        # Initialize components
        self.expander = QueryExpander(self.config)
        self.entity_extractor = EntityExtractor(self.config)
        self.intent_classifier = IntentClassifier(self.config)
        self.normalizer = QueryNormalizer(self.config)
        
        # Metrics
        self.metrics = QueryProcessorMetrics()
        self._metrics_lock = threading.Lock()
        
        logger.info("Enhanced Query Processor initialized")
    
    def process_query(self, query: str) -> ProcessedQuery:
        """Process a query with all enhancements."""
        start_time = time.time()
        
        try:
            # Normalize query
            normalized_query = query
            if self.config.enable_normalization:
                normalized_query = self.normalizer.normalize_query(query)
                with self._metrics_lock:
                    self.metrics.normalization_count += 1
            
            # Expand query
            expanded_terms = []
            if self.config.enable_expansion:
                expanded_terms = self.expander.expand_query(query)
                with self._metrics_lock:
                    self.metrics.expansion_count += 1
            
            # Extract entities
            entities = []
            if self.config.enable_entity_extraction:
                entities = self.entity_extractor.extract_entities(query)
                with self._metrics_lock:
                    self.metrics.entity_extraction_count += 1
                    for entity in entities:
                        self.metrics.entity_distribution[entity.entity_type] += 1
            
            # Classify intent
            intent = QueryIntent.UNKNOWN
            intent_confidence = 0.0
            if self.config.enable_intent_classification:
                intent, intent_confidence = self.intent_classifier.classify_intent(query)
                with self._metrics_lock:
                    self.metrics.intent_classification_count += 1
                    self.metrics.intent_distribution[intent] += 1
            
            # Extract keywords (simple approach)
            keywords = self._extract_keywords(normalized_query)
            
            processing_time = time.time() - start_time
            
            # Update metrics
            with self._metrics_lock:
                self.metrics.total_queries += 1
                self.metrics.total_processing_time += processing_time
            
            return ProcessedQuery(
                original_query=query,
                normalized_query=normalized_query,
                expanded_terms=expanded_terms,
                entities=entities,
                intent=intent,
                intent_confidence=intent_confidence,
                keywords=keywords,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            # Return minimal processed query on error
            return ProcessedQuery(
                original_query=query,
                normalized_query=query,
                expanded_terms=[],
                entities=[],
                intent=QueryIntent.UNKNOWN,
                intent_confidence=0.0,
                keywords=[],
                processing_time=time.time() - start_time
            )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from normalized query."""
        # Simple keyword extraction
        words = query.split()
        # Filter out very short words and common words
        keywords = [word for word in words if len(word) > 2]
        return keywords[:10]  # Limit to top 10 keywords
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        with self._metrics_lock:
            return self.metrics.get_stats()
    
    def reset_stats(self):
        """Reset processor statistics."""
        with self._metrics_lock:
            self.metrics.reset()

# Singleton instance
_query_processor_instance = None
_query_processor_lock = threading.Lock()

def get_query_processor(config: Optional[QueryProcessorConfig] = None) -> EnhancedQueryProcessor:
    """Get singleton query processor instance."""
    global _query_processor_instance
    
    with _query_processor_lock:
        if _query_processor_instance is None:
            _query_processor_instance = EnhancedQueryProcessor(config)
        return _query_processor_instance

def reset_query_processor():
    """Reset singleton query processor instance."""
    global _query_processor_instance
    
    with _query_processor_lock:
        _query_processor_instance = None

# Factory function for easy configuration
def create_query_processor(
    enable_expansion: bool = True,
    enable_entity_extraction: bool = True,
    enable_intent_classification: bool = True,
    enable_normalization: bool = True,
    max_expanded_terms: int = 10
) -> EnhancedQueryProcessor:
    """Create a configured query processor."""
    config = QueryProcessorConfig(
        enable_expansion=enable_expansion,
        enable_entity_extraction=enable_entity_extraction,
        enable_intent_classification=enable_intent_classification,
        enable_normalization=enable_normalization,
        max_expanded_terms=max_expanded_terms
    )
    return EnhancedQueryProcessor(config)

if __name__ == "__main__":
    # Test the query processor
    processor = get_query_processor()
    
    test_queries = [
        "What is artificial intelligence?",
        "How to implement machine learning in Python?",
        "Compare React and Vue.js frameworks",
        "Define neural networks",
        "When was Docker released?",
        "Hello, can you help me?"
    ]
    
    for query in test_queries:
        result = processor.process_query(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {result.intent.value} (confidence: {result.intent_confidence:.2f})")
        print(f"Entities: {[e.text for e in result.entities]}")
        print(f"Expanded terms: {result.expanded_terms}")
        print(f"Keywords: {result.keywords}")
        print(f"Processing time: {result.processing_time:.4f}s")
    
    print(f"\nStats: {processor.get_stats()}") 