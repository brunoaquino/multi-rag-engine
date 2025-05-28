"""
RAG Pipeline for Haystack with Pinecone vector store and Redis caching

This module implements a comprehensive Retrieval-Augmented Generation pipeline
using Haystack 2.0 components, Pinecone for vector storage, Redis caching,
and multi-provider LLM support.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from haystack import Pipeline, Document, component
from haystack.components.embedders import (
    OpenAITextEmbedder, 
    OpenAIDocumentEmbedder,
    SentenceTransformersTextEmbedder,
    SentenceTransformersDocumentEmbedder
)
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.components.converters import TextFileToDocument, MarkdownToDocument, PyPDFToDocument
try:
    from haystack.components.writers.types import DuplicatePolicy
except ImportError:
    # Fallback for different Haystack versions
    try:
        from haystack.components.writers import DuplicatePolicy
    except ImportError:
        # Define our own enum as fallback
        from enum import Enum
        class DuplicatePolicy(Enum):
            NONE = "none"
            SKIP = "skip"
            OVERWRITE = "overwrite"
from haystack.utils import Secret

# Import cache manager and cached components
import sys
sys.path.append('/app')  # Add app root to path
sys.path.append('.')  # Add current dir to path

try:
    from cache_manager import get_cache_manager, CacheManager
    from cached_embedder import CachedOpenAITextEmbedder, CachedOpenAIDocumentEmbedder
    CACHE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Cache components not available: {e}")
    CACHE_AVAILABLE = False

# Import Pinecone connection manager
try:
    from pinecone_manager import get_pinecone_manager, PineconeConfig
    PINECONE_MANAGER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Pinecone manager not available: {e}")
    PINECONE_MANAGER_AVAILABLE = False

# Import optimized chunking components
try:
    # Try importing from pipelines directory first
    from pipelines.optimized_chunking import (
        OptimizedChunker, 
        ParallelDocumentProcessor, 
        ChunkingMetrics,
        create_optimized_chunker,
        create_parallel_processor,
        chunking_metrics
    )
    OPTIMIZED_CHUNKING_AVAILABLE = True
except ImportError:
    try:
        # Fallback to direct import
        from optimized_chunking import (
            OptimizedChunker, 
            ParallelDocumentProcessor, 
            ChunkingMetrics,
            create_optimized_chunker,
            create_parallel_processor,
            chunking_metrics
        )
        OPTIMIZED_CHUNKING_AVAILABLE = True
    except ImportError as e:
        logging.warning(f"Optimized chunking not available: {e}")
        OPTIMIZED_CHUNKING_AVAILABLE = False

# Pinecone integration
try:
    from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
    from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

# Anthropic fallback generator (custom implementation needed)
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Import re-ranking manager
try:
    from reranking_manager import get_reranking_manager, ReRankingManager, ReRankingConfig
    RERANKING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Re-ranking manager not available: {e}")
    RERANKING_AVAILABLE = False

# Import query processor and load balancer
try:
    from query_processor import get_query_processor, EnhancedQueryProcessor, QueryProcessorConfig
    QUERY_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Query processor not available: {e}")
    QUERY_PROCESSOR_AVAILABLE = False

try:
    from load_balancer import get_load_balancer, LoadBalancer, LoadBalancerConfig, QueryRequest, QueryPriority
    LOAD_BALANCER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Load balancer not available: {e}")
    LOAD_BALANCER_AVAILABLE = False

# Import model manager
try:
    from model_manager import (
        get_model_manager, initialize_model_manager, ModelManager, 
        ProviderConfig, ProviderType, GenerationRequest, Priority, RetryStrategy
    )
    MODEL_MANAGER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Model manager not available: {e}")
    MODEL_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    # Pinecone settings
    pinecone_index: str = "haystack-rag"
    pinecone_dimension: int = 1536  # OpenAI ada-002 dimension
    pinecone_metric: str = "cosine"
    pinecone_region: str = "us-east-1"
    pinecone_cloud: str = "aws"
    
    # Embedding settings
    embedding_model: str = "text-embedding-ada-002"  # or "sentence-transformers/all-MiniLM-L6-v2"
    use_openai_embeddings: bool = True
    
    # LLM settings
    primary_llm: str = "gpt-4o-mini"
    fallback_llm: str = "claude-3-haiku-20240307"
    max_tokens: int = 1000
    temperature: float = 0.7
    
    # Document processing
    chunk_size: int = 512
    chunk_overlap: int = 50
    split_by: str = "word"  # or "sentence", "passage"
    
    # Optimized chunking settings
    use_optimized_chunking: bool = True
    enable_parallel_processing: bool = True
    parallel_workers: int = 4
    parallel_batch_size: int = 50
    
    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float = 0.7
    
    # Cache settings
    enable_cache: bool = True
    cache_query_results: bool = True
    cache_embeddings: bool = True
    
    # Pinecone optimization settings
    enable_pinecone_optimization: bool = True
    pinecone_cache_ttl: int = 3600  # 1 hour
    max_cached_stores: int = 50
    
    # Re-ranking settings
    enable_reranking: bool = True
    enable_cross_encoder: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    reranking_top_k: int = 20  # Documents to re-rank with cross-encoder
    semantic_weight: float = 0.6
    bm25_weight: float = 0.2
    metadata_weight: float = 0.1
    length_weight: float = 0.05
    recency_weight: float = 0.05
    
    # Query processing settings
    enable_query_processing: bool = True
    enable_query_expansion: bool = True
    enable_entity_extraction: bool = True
    enable_intent_classification: bool = True
    enable_query_normalization: bool = True
    max_expanded_terms: int = 10
    
    # Load balancing settings
    enable_load_balancing: bool = True
    max_concurrent_queries: int = 10
    query_timeout: float = 30.0
    load_balancing_strategy: str = "least_connections"  # round_robin, least_connections, weighted_round_robin, random
    circuit_breaker_threshold: int = 5
    max_queue_size: int = 100
    
    # Multi-model management settings
    enable_model_manager: bool = True
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    default_fallback_chain: str = "default"  # default, high_capability, cost_optimized
    enable_cost_optimization: bool = True
    openai_rate_limit: int = 10000  # tokens per minute
    anthropic_rate_limit: int = 5000  # tokens per minute
    
    # RAG mode configuration
    rag_mode: str = "hybrid"  # "strict", "hybrid", or "enhanced"
    
    # Prompt template
    rag_prompt_template: str = """
Você é um assistente de IA útil que responde perguntas baseado no contexto fornecido.
Use o contexto abaixo para responder à pergunta de forma precisa e abrangente.

SEMPRE RESPONDA EM PORTUGUÊS BRASILEIRO.

Contexto:
{% for document in documents %}
{{ document.content }}
{% if not loop.last %}
---
{% endif %}
{% endfor %}

Pergunta: {{ question }}

Instruções:
- Base sua resposta principalmente no contexto fornecido
- Se o contexto não contém informações suficientes, declare isso claramente
- Seja específico e cite detalhes relevantes do contexto
- Se não tiver certeza, expresse essa incerteza
- Mantenha sua resposta focada e relevante à pergunta
- SEMPRE responda em português brasileiro

Resposta:
"""

    # Enhanced hybrid RAG template that combines context with global knowledge
    hybrid_rag_prompt_template: str = """
Você é um assistente de IA inteligente com acesso tanto ao contexto específico de documentos quanto ao conhecimento geral.

SEMPRE RESPONDA EM PORTUGUÊS BRASILEIRO.

CONTEXTO DOS DOCUMENTOS:
{% for document in documents %}
Documento {{ loop.index }}:
{{ document.content }}
{% if not loop.last %}

---

{% endif %}
{% endfor %}

PERGUNTA: {{ question }}

INSTRUÇÕES:
1. **Foco Primário**: Use o contexto dos documentos fornecidos como sua principal fonte de informação
2. **Integração de Contexto**: Quando os documentos contêm informações relevantes, cite e referencie-os especificamente
3. **Enriquecimento de Conhecimento**: Se o contexto dos documentos estiver incompleto ou se beneficiaria de contexto adicional, complemente com seu conhecimento geral para fornecer uma resposta mais abrangente
4. **Atribuição Clara**: 
   - Ao usar informações de documentos, indique isso claramente (ex: "De acordo com os documentos fornecidos...")
   - Ao adicionar conhecimento geral, faça essa distinção (ex: "Adicionalmente, do conhecimento geral...")
5. **Resposta Equilibrada**: Procure respostas que sejam baseadas no contexto fornecido e úteis de forma completa
6. **Transparência**: Se os documentos não tiverem informações sobre aspectos importantes da pergunta, reconheça isso e forneça orientações gerais quando apropriado
7. **IDIOMA**: SEMPRE responda em português brasileiro, independentemente do idioma da pergunta ou documentos

FORMATO DA RESPOSTA:
- Comece com informações dos documentos quando disponíveis
- Enriqueça com conhecimento geral relevante para tornar a resposta mais completa e útil
- Distinga claramente entre conhecimento baseado em documentos e conhecimento geral
- Termine com quaisquer ressalvas ou limitações importantes
- Use sempre português brasileiro

RESPOSTA:
"""

    # Strict RAG template (original behavior)
    strict_rag_prompt_template: str = """
Você é um assistente de IA baseado em documentos. Você deve responder perguntas APENAS usando informações do contexto fornecido.

SEMPRE RESPONDA EM PORTUGUÊS BRASILEIRO.

CONTEXTO:
{% for document in documents %}
{{ document.content }}
{% if not loop.last %}
---
{% endif %}
{% endfor %}

PERGUNTA: {{ question }}

INSTRUÇÕES RIGOROSAS:
- Responda APENAS usando informações explicitamente declaradas no contexto fornecido
- Se o contexto não contém as informações necessárias para responder à pergunta, responda com "Não posso responder a esta pergunta com base nos documentos fornecidos."
- NÃO use conhecimento externo ou faça suposições além do que está no contexto
- Cite ou referencie partes específicas do contexto quando possível
- Seja preciso e factual
- SEMPRE responda em português brasileiro, independentemente do idioma da pergunta ou documentos

RESPOSTA:
"""


class FallbackAnthropicGenerator:
    """Fallback generator using Anthropic Claude when OpenAI fails"""
    
    def __init__(self, model: str = "claude-3-haiku-20240307", max_tokens: int = 1000):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not available")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
    
    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str) -> Dict[str, Any]:
        """Run Anthropic generation"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            generated_text = response.content[0].text
            
            return {
                "replies": [generated_text],
                "meta": [{"model": self.model, "provider": "anthropic"}]
            }
        except Exception as e:
            logger.error(f"Anthropic fallback failed: {e}")
            return {
                "replies": ["Desculpe, não foi possível processar sua solicitação no momento."],
                "meta": [{"error": str(e), "provider": "anthropic_fallback"}]
            }


class ModelManagerGenerator:
    """Generator that uses ModelManager for multi-model support with fallback"""
    
    def __init__(
        self, 
        model_manager: Optional[ModelManager] = None,
        fallback_chain: str = "default",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        priority: Priority = Priority.NORMAL
    ):
        self.model_manager = model_manager or get_model_manager()
        self.fallback_chain = fallback_chain
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.priority = priority
        self.fallback_generator = None
        
        # Initialize fallback generator if model manager not available
        if not self.model_manager:
            try:
                self.fallback_generator = FallbackAnthropicGenerator(max_tokens=max_tokens)
                logger.warning("ModelManager not available, using fallback Anthropic generator")
            except Exception as e:
                logger.error(f"Failed to initialize fallback generator: {e}")
    
    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str) -> Dict[str, Any]:
        """Run generation using ModelManager"""
        # Use fallback if model manager not available
        if not self.model_manager:
            if self.fallback_generator:
                return self.fallback_generator.run(prompt)
            else:
                return {
                    "replies": ["Desculpe, o serviço de geração não está disponível no momento."],
                    "meta": [{"error": "No generation service available"}]
                }
        
        # Create generation request
        request = GenerationRequest(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            fallback_chain=self.fallback_chain,
            priority=self.priority,
            metadata={"timestamp": time.time()}
        )
        
        try:
            # Use asyncio.run for async method
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(self.model_manager.generate(request))
            finally:
                loop.close()
            
            return {
                "replies": [response.text],
                "meta": [{
                    "model": response.model_used,
                    "provider": response.provider_used.value,
                    "tokens_used": response.tokens_used,
                    "cost_usd": response.cost_usd,
                    "latency_ms": response.latency_ms,
                    "fallback_used": response.fallback_used,
                    "retries_attempted": response.retries_attempted
                }]
            }
            
        except Exception as e:
            logger.error(f"ModelManager generation failed: {e}")
            
            # Try fallback generator as last resort
            if self.fallback_generator:
                logger.info("Attempting fallback to Anthropic generator")
                try:
                    result = self.fallback_generator.run(prompt)
                    # Add fallback indicator to meta
                    if result["meta"]:
                        result["meta"][0]["emergency_fallback"] = True
                    return result
                except Exception as fallback_error:
                    logger.error(f"Emergency fallback also failed: {fallback_error}")
            
            return {
                "replies": ["Desculpe, não foi possível processar sua solicitação no momento. Tente novamente mais tarde."],
                "meta": [{"error": str(e), "emergency_fallback_failed": True}]
            }


class RAGPipeline:
    """
    Comprehensive RAG pipeline with Pinecone vector store
    
    Features:
    - Document indexing with preprocessing
    - Semantic search with Pinecone
    - Multi-provider LLM generation (OpenAI + Anthropic fallback)
    - Configurable chunking and embedding strategies
    - Metadata filtering and namespace support
    """
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.document_store = None
        self.indexing_pipeline = None
        self.query_pipeline = None
        self.cache_manager = None
        self.optimized_chunker = None
        self.parallel_processor = None
        self.pinecone_manager = None
        self.reranking_manager = None
        self.query_processor = None
        self.load_balancer = None
        self.model_manager = None
        
        # Initialize cache manager
        if self.config.enable_cache and CACHE_AVAILABLE:
            self.cache_manager = get_cache_manager()
            logger.info("Cache manager initialized for RAG pipeline")
        else:
            logger.info("Cache disabled or not available for RAG pipeline")
        
        # Initialize optimized chunking components
        if self.config.use_optimized_chunking and OPTIMIZED_CHUNKING_AVAILABLE:
            self.optimized_chunker = create_optimized_chunker()
            if self.config.enable_parallel_processing:
                self.parallel_processor = create_parallel_processor(
                    max_workers=self.config.parallel_workers,
                    batch_size=self.config.parallel_batch_size
                )
            logger.info("Optimized chunking components initialized")
        else:
            logger.info("Using standard chunking strategy")
        
        # Initialize Pinecone connection manager
        if self.config.enable_pinecone_optimization and PINECONE_MANAGER_AVAILABLE:
            pinecone_config = PineconeConfig(
                index=self.config.pinecone_index,
                dimension=self.config.pinecone_dimension,
                metric=self.config.pinecone_metric,
                region=self.config.pinecone_region,
                cloud=self.config.pinecone_cloud,
                default_top_k=self.config.top_k,
                cache_ttl=self.config.pinecone_cache_ttl,
                max_cached_stores=self.config.max_cached_stores
            )
            self.pinecone_manager = get_pinecone_manager(pinecone_config)
            logger.info("Pinecone connection manager initialized")
        else:
            logger.info("Using standard Pinecone operations")
        
        # Initialize re-ranking manager
        if self.config.enable_reranking and RERANKING_AVAILABLE:
            try:
                reranking_config = ReRankingConfig(
                    enable_cross_encoder=self.config.enable_cross_encoder,
                    cross_encoder_model=self.config.cross_encoder_model,
                    cross_encoder_top_k=self.config.reranking_top_k,
                    semantic_weight=self.config.semantic_weight,
                    bm25_weight=self.config.bm25_weight,
                    metadata_weight=self.config.metadata_weight,
                    length_weight=self.config.length_weight,
                    recency_weight=self.config.recency_weight
                )
                self.reranking_manager = get_reranking_manager(reranking_config)
                logger.info("Re-ranking manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize re-ranking manager: {e}")
        else:
            logger.info("Re-ranking not enabled or not available")
        
        # Initialize query processor
        if self.config.enable_query_processing and QUERY_PROCESSOR_AVAILABLE:
            try:
                query_config = QueryProcessorConfig(
                    enable_expansion=self.config.enable_query_expansion,
                    enable_entity_extraction=self.config.enable_entity_extraction,
                    enable_intent_classification=self.config.enable_intent_classification,
                    enable_normalization=self.config.enable_query_normalization,
                    max_expanded_terms=self.config.max_expanded_terms
                )
                self.query_processor = get_query_processor(query_config)
                logger.info("Enhanced query processor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize query processor: {e}")
                self.query_processor = None
        else:
            logger.info("Query processing disabled or not available")
        
        # Initialize load balancer
        if self.config.enable_load_balancing and LOAD_BALANCER_AVAILABLE:
            try:
                from load_balancer import LoadBalancingStrategy
                
                # Map string to enum
                strategy_mapping = {
                    "round_robin": LoadBalancingStrategy.ROUND_ROBIN,
                    "least_connections": LoadBalancingStrategy.LEAST_CONNECTIONS,
                    "weighted_round_robin": LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN,
                    "random": LoadBalancingStrategy.RANDOM
                }
                
                lb_config = LoadBalancerConfig(
                    max_queue_size=self.config.max_queue_size,
                    max_workers=self.config.max_concurrent_queries,
                    worker_timeout=self.config.query_timeout,
                    strategy=strategy_mapping.get(self.config.load_balancing_strategy, LoadBalancingStrategy.LEAST_CONNECTIONS),
                    failure_threshold=self.config.circuit_breaker_threshold
                )
                self.load_balancer = get_load_balancer(lb_config)
                
                # Add RAG processing worker
                self.load_balancer.add_worker(
                    worker_id="rag_worker_1",
                    name="RAG Pipeline Worker",
                    processor_func=self._process_rag_query,
                    max_concurrent=self.config.max_concurrent_queries
                )
                
                # Start load balancer
                self.load_balancer.start()
                logger.info("Load balancer initialized and started")
            except Exception as e:
                logger.error(f"Failed to initialize load balancer: {e}")
                self.load_balancer = None
        else:
            logger.info("Load balancing disabled or not available")
        
        # Initialize model manager
        if self.config.enable_model_manager and MODEL_MANAGER_AVAILABLE:
            try:
                # Prepare provider configurations
                provider_configs = []
                
                # OpenAI configuration
                openai_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
                if openai_key:
                    openai_config = ProviderConfig(
                        provider_type=ProviderType.OPENAI,
                        api_key=openai_key,
                        rate_limit_tokens_per_minute=self.config.openai_rate_limit,
                        retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                        max_retries=3,
                        timeout_seconds=30.0
                    )
                    provider_configs.append(openai_config)
                
                # Anthropic configuration
                anthropic_key = self.config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
                if anthropic_key:
                    anthropic_config = ProviderConfig(
                        provider_type=ProviderType.ANTHROPIC,
                        api_key=anthropic_key,
                        rate_limit_tokens_per_minute=self.config.anthropic_rate_limit,
                        retry_strategy=RetryStrategy.IMMEDIATE_FALLBACK,
                        max_retries=2,
                        timeout_seconds=30.0
                    )
                    provider_configs.append(anthropic_config)
                
                if provider_configs:
                    self.model_manager = initialize_model_manager(provider_configs)
                    logger.info(f"Model manager initialized with {len(provider_configs)} providers")
                else:
                    logger.warning("No API keys found for model manager providers")
                    
            except Exception as e:
                logger.error(f"Failed to initialize model manager: {e}")
                self.model_manager = None
        else:
            logger.info("Model manager disabled or not available")
        
        # Initialize document store
        self._init_document_store()
        
        # Create pipelines
        self._create_indexing_pipeline()
        self._create_query_pipeline()
    
    def _get_prompt_template(self, mode: Optional[str] = None) -> str:
        """Get the appropriate prompt template based on RAG mode"""
        selected_mode = mode or self.config.rag_mode
        
        if selected_mode == "strict":
            return self.config.strict_rag_prompt_template
        elif selected_mode == "hybrid":
            return self.config.hybrid_rag_prompt_template
        elif selected_mode == "enhanced":
            # For future: more advanced hybrid mode
            return self.config.hybrid_rag_prompt_template
        else:
            # Default fallback
            logger.warning(f"Unknown RAG mode '{selected_mode}', using hybrid mode")
            return self.config.hybrid_rag_prompt_template
    
    def _init_document_store(self):
        """Initialize Pinecone document store"""
        if not PINECONE_AVAILABLE:
            raise ImportError("pinecone-haystack package not available. Install with: pip install pinecone-haystack")
        
        # Use optimized connection manager if available
        if self.pinecone_manager:
            try:
                self.document_store = self.pinecone_manager.get_document_store()
                logger.info(f"Initialized optimized Pinecone document store with index: {self.config.pinecone_index}")
                return
            except Exception as e:
                logger.warning(f"Failed to use optimized Pinecone manager, falling back to standard: {e}")
        
        # Fallback to standard initialization
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        try:
            self.document_store = PineconeDocumentStore(
                index=self.config.pinecone_index,
                metric=self.config.pinecone_metric,
                dimension=self.config.pinecone_dimension,
                spec={
                    "serverless": {
                        "region": self.config.pinecone_region,
                        "cloud": self.config.pinecone_cloud
                    }
                }
            )
            logger.info(f"Initialized standard Pinecone document store with index: {self.config.pinecone_index}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone document store: {e}")
            raise
    
    def _create_indexing_pipeline(self):
        """Create document indexing pipeline with optimized chunking"""
        self.indexing_pipeline = Pipeline()
        
        # Document cleaning
        cleaner = DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
            remove_repeated_substrings=False
        )
        
        # Document splitting (fallback or standard splitting)
        if not self.config.use_optimized_chunking or not OPTIMIZED_CHUNKING_AVAILABLE:
        splitter = DocumentSplitter(
            split_by=self.config.split_by,
            split_length=self.config.chunk_size,
            split_overlap=self.config.chunk_overlap
        )
        else:
            # Use optimized chunking - will be handled in index_documents method
            splitter = None
        
        # Document embedder (with cache if available)
        if self.config.use_openai_embeddings:
            if self.config.cache_embeddings and self.cache_manager and CACHE_AVAILABLE:
                embedder = CachedOpenAIDocumentEmbedder(
                    model=self.config.embedding_model,
                    api_key=Secret.from_env_var("OPENAI_API_KEY"),
                    cache_manager=self.cache_manager
                )
                logger.info("Using cached OpenAI document embedder")
            else:
                embedder = OpenAIDocumentEmbedder(
                    model=self.config.embedding_model,
                    api_key=Secret.from_env_var("OPENAI_API_KEY")
                )
                logger.info("Using standard OpenAI document embedder")
        else:
            embedder = SentenceTransformersDocumentEmbedder(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Using SentenceTransformers document embedder")
        
        # Document writer
        writer = DocumentWriter(document_store=self.document_store)
        
        # Add components based on chunking strategy
        self.indexing_pipeline.add_component("cleaner", cleaner)
        
        if splitter:  # Standard chunking
        self.indexing_pipeline.add_component("splitter", splitter)
        self.indexing_pipeline.add_component("embedder", embedder)
        self.indexing_pipeline.add_component("writer", writer)
        
        # Connect components
        self.indexing_pipeline.connect("cleaner", "splitter")
        self.indexing_pipeline.connect("splitter", "embedder")
        self.indexing_pipeline.connect("embedder", "writer")
        
            logger.info("Created standard indexing pipeline")
        else:  # Optimized chunking - simpler pipeline
            self.indexing_pipeline.add_component("embedder", embedder)
            self.indexing_pipeline.add_component("writer", writer)
            
            # Connect components (chunking handled separately)
            self.indexing_pipeline.connect("cleaner", "embedder")
            self.indexing_pipeline.connect("embedder", "writer")
            
            logger.info("Created optimized indexing pipeline")
    
    def _create_query_pipeline(self):
        """Create query processing pipeline with fallback support"""
        self.query_pipeline = Pipeline()
        
        # Text embedder for queries (with cache if available)
        if self.config.use_openai_embeddings:
            if self.config.cache_embeddings and self.cache_manager and CACHE_AVAILABLE:
                text_embedder = CachedOpenAITextEmbedder(
                    model=self.config.embedding_model,
                    api_key=Secret.from_env_var("OPENAI_API_KEY"),
                    cache_manager=self.cache_manager
                )
                logger.info("Using cached OpenAI text embedder")
            else:
                text_embedder = OpenAITextEmbedder(
                    model=self.config.embedding_model,
                    api_key=Secret.from_env_var("OPENAI_API_KEY")
                )
                logger.info("Using standard OpenAI text embedder")
        else:
            text_embedder = SentenceTransformersTextEmbedder(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Using SentenceTransformers text embedder")
        
        # Retriever (use optimized if available)
        if self.pinecone_manager:
            retriever = self.pinecone_manager.get_retriever(top_k=self.config.top_k)
            logger.info("Using optimized Pinecone retriever")
        else:
        retriever = PineconeEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.config.top_k
        )
            logger.info("Using standard Pinecone retriever")
        
        # Prompt builder (using dynamic template based on mode)
        prompt_template = self._get_prompt_template()
        prompt_builder = PromptBuilder(template=prompt_template)
        logger.info(f"Using RAG mode: {self.config.rag_mode}")
        
        # Primary LLM generator
        generator = ModelManagerGenerator(
            model_manager=self.model_manager,
            fallback_chain=self.config.default_fallback_chain,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        # Add components
        self.query_pipeline.add_component("text_embedder", text_embedder)
        self.query_pipeline.add_component("retriever", retriever)
        self.query_pipeline.add_component("prompt_builder", prompt_builder)
        self.query_pipeline.add_component("generator", generator)
        
        # Connect components
        self.query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.query_pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.query_pipeline.connect("prompt_builder", "generator")
        
        logger.info("Created query pipeline")
    
    def index_documents(
        self, 
        documents: List[Document], 
        batch_size: int = 100,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Index documents into the vector store with optimized chunking and namespace support
        
        Args:
            documents: List of Haystack Document objects
            batch_size: Number of documents to process in each batch
            namespace: Pinecone namespace for document indexing
            
        Returns:
            Dictionary with indexing results
        """
        try:
            import time
            start_time = time.time()
            
            # Apply optimized chunking if enabled
            if self.config.use_optimized_chunking and OPTIMIZED_CHUNKING_AVAILABLE:
                logger.info(f"Applying optimized chunking to {len(documents)} documents...")
                
                if self.config.enable_parallel_processing and self.parallel_processor:
                    # Use parallel processing
                    processed_documents = self.parallel_processor.process_documents_sync(documents)
                else:
                    # Use optimized chunking sequentially
                    processed_documents = []
                    for doc in documents:
                        chunked_docs = self.optimized_chunker.chunk_document(doc)
                        processed_documents.extend(chunked_docs)
                
                chunking_time = time.time() - start_time
                logger.info(f"Optimized chunking completed: {len(documents)} docs → {len(processed_documents)} chunks in {chunking_time:.2f}s")
                
                # Record metrics
                if OPTIMIZED_CHUNKING_AVAILABLE:
                    chunking_metrics.record_chunking(documents, processed_documents, chunking_time)
                
                documents = processed_documents
            
            # Process documents in batches
            total_docs = len(documents)
            indexed_count = 0
            
            # For Pinecone namespace support, we need to create a namespace-specific document store
            if namespace:
                # Use optimized connection manager if available
                if self.pinecone_manager:
                    namespace_doc_store = self.pinecone_manager.get_document_store(namespace)
                    logger.debug(f"Using optimized document store for namespace: {namespace}")
                else:
                    # Create a namespace-specific document store for indexing
                    namespace_doc_store = PineconeDocumentStore(
                        index=self.config.pinecone_index,
                        namespace=namespace,  # Set namespace at document store level
                        metric=self.config.pinecone_metric,
                        dimension=self.config.pinecone_dimension,
                        spec={
                            "serverless": {
                                "region": self.config.pinecone_region,
                                "cloud": self.config.pinecone_cloud
                            }
                        }
                    )
                    logger.debug(f"Created standard document store for namespace: {namespace}")
                
                # Create namespace-specific writer
                namespace_writer = DocumentWriter(
                    document_store=namespace_doc_store,
                    policy=DuplicatePolicy.OVERWRITE
                )
                
                # Create a temporary pipeline for this namespace
                temp_pipeline = Pipeline()
                temp_pipeline.add_component("cleaner", self.indexing_pipeline.get_component("cleaner"))
                temp_pipeline.add_component("splitter", self.indexing_pipeline.get_component("splitter"))
                temp_pipeline.add_component("embedder", self.indexing_pipeline.get_component("embedder"))
                temp_pipeline.add_component("writer", namespace_writer)
                
                # Connect components
                temp_pipeline.connect("cleaner", "splitter")
                temp_pipeline.connect("splitter", "embedder")
                temp_pipeline.connect("embedder", "writer")
                
                pipeline_to_use = temp_pipeline
            else:
                pipeline_to_use = self.indexing_pipeline
            
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                
                # If namespace provided, ensure each document has it in metadata
                if namespace:
                    for doc in batch:
                        if doc.meta is None:
                            doc.meta = {}
                        doc.meta["namespace"] = namespace
                
                result = pipeline_to_use.run({
                    "cleaner": {"documents": batch}
                })
                
                indexed_count += len(batch)
                namespace_info = f" in namespace '{namespace}'" if namespace else ""
                logger.info(f"Indexed batch {i//batch_size + 1}: {indexed_count}/{total_docs} documents{namespace_info}")
            
            return {
                "success": True,
                "indexed_documents": indexed_count,
                "total_documents": total_docs,
                "namespace": namespace
            }
            
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "indexed_documents": 0,
                "namespace": namespace
            }
    
    def index_files(
        self, 
        file_paths: List[str], 
        file_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Index files from disk
        
        Args:
            file_paths: List of file paths to index
            file_type: Type of files ('text', 'markdown', 'pdf', 'auto')
            
        Returns:
            Dictionary with indexing results
        """
        documents = []
        
        for file_path in file_paths:
            try:
                # Auto-detect file type if needed
                if file_type == "auto":
                    if file_path.endswith(('.md', '.markdown')):
                        converter = MarkdownToDocument()
                    elif file_path.endswith('.pdf'):
                        converter = PyPDFToDocument()
                    else:
                        converter = TextFileToDocument()
                else:
                    # Use specified converter
                    if file_type == "markdown":
                        converter = MarkdownToDocument()
                    elif file_type == "pdf":
                        converter = PyPDFToDocument()
                    else:
                        converter = TextFileToDocument()
                
                # Convert file to documents
                result = converter.run(sources=[file_path])
                documents.extend(result["documents"])
                
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                continue
        
        if not documents:
            return {
                "success": False,
                "error": "No documents were successfully processed",
                "indexed_documents": 0
            }
        
        return self.index_documents(documents)
    
    def _process_rag_query(self, query_data: str) -> Dict[str, Any]:
        """Internal method for processing RAG queries through load balancer."""
        try:
            # Parse query data (assuming JSON string)
            import json
            data = json.loads(query_data) if isinstance(query_data, str) else {"question": query_data}
            
            question = data.get("question", query_data)
            filters = data.get("filters")
            namespace = data.get("namespace")
            rag_mode = data.get("rag_mode")
            
            # Process query directly without load balancer
            return self._execute_query(question, filters, namespace, rag_mode)
            
        except Exception as e:
            logger.error(f"Error in _process_rag_query: {e}")
            return {
                "answer": f"Erro no processamento da query: {str(e)}",
                "source_documents": [],
                "meta": {"error": str(e)},
                "provider_used": "error",
                "rag_mode": "error"
            }
    
    def query(
        self, 
        question: str, 
        filters: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        rag_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline with enhanced processing, caching, and load balancing support
        
        Args:
            question: User question
            filters: Metadata filters for retrieval
            namespace: Pinecone namespace for retrieval
            rag_mode: RAG mode override ("strict", "hybrid", or "enhanced")
            
        Returns:
            Dictionary with query results
        """
        # Enhanced query processing
        processed_query = None
        enhanced_question = question
        
        if self.query_processor:
            try:
                processed_query = self.query_processor.process_query(question)
                
                # Use expanded query if available
                if processed_query.expanded_terms:
                    # Combine original question with expanded terms
                    expanded_terms_str = " ".join(processed_query.expanded_terms[:5])  # Limit to top 5
                    enhanced_question = f"{question} {expanded_terms_str}"
                    logger.debug(f"Enhanced query with expanded terms: {enhanced_question[:100]}...")
                
                logger.debug(f"Query processed - Intent: {processed_query.intent.value}, "
                           f"Entities: {len(processed_query.entities)}, "
                           f"Expanded terms: {len(processed_query.expanded_terms)}")
                           
            except Exception as e:
                logger.warning(f"Query processing failed, using original query: {e}")
        
        # Determine effective RAG mode
        effective_rag_mode = rag_mode or self.config.rag_mode
        
        # Check cache first if enabled (include rag_mode in cache key)
        query_embedding = None
        if self.config.cache_query_results and self.cache_manager and CACHE_AVAILABLE:
            # First try semantic cache which includes exact match fallback
            try:
                # Generate embedding for semantic matching
                embedding_result = self.query_pipeline.get_component("text_embedder").run(text=question)
                query_embedding = embedding_result["embedding"]
                
                cached_result = self.cache_manager.get_semantic_query_cache(
                    question,
                    query_embedding,
                    filters=filters, 
                    namespace=namespace,
                    model=self.config.primary_llm,
                    extra_key=f"mode_{effective_rag_mode}"
                )
            except Exception as embedding_error:
                logger.warning(f"Failed to generate embedding for semantic cache: {embedding_error}")
                # Fallback to regular cache
            cached_result = self.cache_manager.get_query_cache(
                question, 
                filters=filters, 
                namespace=namespace,
                    model=self.config.primary_llm,
                    extra_key=f"mode_{effective_rag_mode}"
            )
            
            if cached_result:
                logger.debug(f"Cache HIT for query: {question[:50]}... (mode: {effective_rag_mode})")
                cached_result["result"]["meta"]["cached"] = True
                cached_result["result"]["meta"]["rag_mode"] = effective_rag_mode
                return cached_result["result"]
            
            logger.debug(f"Cache MISS for query: {question[:50]}... (mode: {effective_rag_mode})")
        
        try:
            # Prepare query parameters
            query_params = {
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question}
            }
            
            # Determine which pipeline to use
            pipeline_to_use = self.query_pipeline
            
            # If rag_mode is different from default, create temporary pipeline
            if effective_rag_mode != self.config.rag_mode:
                logger.info(f"Creating temporary pipeline for mode: {effective_rag_mode}")
                
                # Create temporary prompt builder with different template
                temp_prompt_template = self._get_prompt_template(effective_rag_mode)
                temp_prompt_builder = PromptBuilder(template=temp_prompt_template)
                
                # Create temporary pipeline
                temp_pipeline = Pipeline()
                temp_pipeline.add_component("text_embedder", self.query_pipeline.get_component("text_embedder"))
                temp_pipeline.add_component("retriever", self.query_pipeline.get_component("retriever"))
                temp_pipeline.add_component("prompt_builder", temp_prompt_builder)
                temp_pipeline.add_component("generator", self.query_pipeline.get_component("generator"))
                
                # Connect components
                temp_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
                temp_pipeline.connect("retriever.documents", "prompt_builder.documents")
                temp_pipeline.connect("prompt_builder", "generator")
                
                pipeline_to_use = temp_pipeline
            
            # Add filters if provided
            if filters:
                query_params["retriever"] = {"filters": filters}
            
            # For namespace queries, create a temporary retriever with namespace-specific document store
            if namespace:
                # Use optimized connection manager if available
                if self.pinecone_manager:
                    namespace_retriever = self.pinecone_manager.get_retriever(
                        namespace=namespace, 
                        top_k=self.config.top_k
                    )
                    logger.debug(f"Using optimized retriever for namespace: {namespace}")
                else:
                    # Create namespace-specific document store for retrieval
                    namespace_doc_store = PineconeDocumentStore(
                        index=self.config.pinecone_index,
                        namespace=namespace,
                        metric=self.config.pinecone_metric,
                        dimension=self.config.pinecone_dimension,
                        spec={
                            "serverless": {
                                "region": self.config.pinecone_region,
                                "cloud": self.config.pinecone_cloud
                            }
                        }
                    )
                    
                    # Create namespace-specific retriever
                    namespace_retriever = PineconeEmbeddingRetriever(
                        document_store=namespace_doc_store,
                        top_k=self.config.top_k
                    )
                    logger.debug(f"Created standard retriever for namespace: {namespace}")
                
                # Create temporary pipeline with namespace-specific retriever
                temp_query_pipeline = Pipeline()
                temp_query_pipeline.add_component("text_embedder", self.query_pipeline.get_component("text_embedder"))
                temp_query_pipeline.add_component("retriever", namespace_retriever)
                temp_query_pipeline.add_component("prompt_builder", self.query_pipeline.get_component("prompt_builder"))
                temp_query_pipeline.add_component("generator", self.query_pipeline.get_component("generator"))
                
                # Connect components
                temp_query_pipeline.connect("text_embedder", "retriever")
                temp_query_pipeline.connect("retriever", "prompt_builder.documents")
                temp_query_pipeline.connect("prompt_builder", "generator")
                
                # Use the temporary pipeline for this query
                pipeline_to_use = temp_query_pipeline
                # Remove retriever params since it's now namespace-specific
                if "retriever" in query_params:
                    # Keep filters but apply them to the new retriever
                    if "filters" in query_params["retriever"]:
                        query_params["retriever"] = {"filters": query_params["retriever"]["filters"]}
                    else:
                        del query_params["retriever"]
            else:
                pipeline_to_use = self.query_pipeline
            
            # Run primary pipeline (or namespace-specific pipeline)
            try:
                result = pipeline_to_use.run(query_params)
                
                # Handle both string and object replies with better error handling
                reply = result["generator"]["replies"][0]
                if hasattr(reply, 'content'):
                    answer = reply.content
                elif hasattr(reply, 'text'):
                    answer = reply.text
                elif isinstance(reply, str):
                    answer = reply
                else:
                    # Try to get content from dict-like object
                    if isinstance(reply, dict) and 'content' in reply:
                        answer = reply['content']
                    elif isinstance(reply, dict) and 'text' in reply:
                        answer = reply['text']
                    else:
                        answer = str(reply)
                
                # Apply re-ranking if enabled
                retrieved_documents = result["retriever"]["documents"]
                if self.config.enable_reranking and self.reranking_manager and retrieved_documents:
                    try:
                        reranked_documents = self.reranking_manager.rerank_documents(
                            query=question,
                            documents=retrieved_documents
                        )
                        logger.debug(f"Re-ranked {len(retrieved_documents)} documents")
                    except Exception as e:
                        logger.warning(f"Re-ranking failed, using original documents: {e}")
                        reranked_documents = retrieved_documents
                else:
                    reranked_documents = retrieved_documents
                
                response = {
                    "answer": answer,
                    "source_documents": reranked_documents,
                    "meta": result["generator"].get("meta", {}),
                    "provider_used": "openai",
                    "rag_mode": effective_rag_mode
                }
                
                # Cache the result if enabled
                if self.config.cache_query_results and self.cache_manager and CACHE_AVAILABLE:
                    # Use semantic caching if we have the embedding
                    if query_embedding:
                        self.cache_manager.set_semantic_query_cache(
                            question,
                            query_embedding,
                            response,
                            filters=filters, 
                            namespace=namespace,
                            model=self.config.primary_llm,
                            extra_key=f"mode_{effective_rag_mode}"
                        )
                    else:
                        # Fallback to regular cache
                    self.cache_manager.set_query_cache(
                        question, 
                        response,
                        filters=filters, 
                        namespace=namespace,
                            model=self.config.primary_llm,
                            extra_key=f"mode_{effective_rag_mode}"
                    )
                
                return response
                
            except Exception as primary_error:
                logger.warning(f"Primary LLM failed: {primary_error}")
                logger.debug(f"Result structure: {result if 'result' in locals() else 'No result available'}")
                logger.debug(f"Query params: {query_params}")
                
                # Try fallback with Anthropic
                if ANTHROPIC_AVAILABLE:
                    try:
                        # Get retrieval results manually
                        embedding_result = pipeline_to_use.get_component("text_embedder").run(text=question)
                        
                        retrieval_result = pipeline_to_use.get_component("retriever").run(
                            query_embedding=embedding_result["embedding"],
                            filters=filters
                        )
                        prompt_result = pipeline_to_use.get_component("prompt_builder").run(
                            question=question,
                            documents=retrieval_result["documents"]
                        )
                        
                        # Use fallback generator
                        fallback_generator = FallbackAnthropicGenerator(
                            model=self.config.fallback_llm,
                            max_tokens=self.config.max_tokens
                        )
                        
                        generation_result = fallback_generator.run(prompt_result["prompt"])
                        
                        # Apply re-ranking for fallback as well
                        fallback_documents = retrieval_result["documents"]
                        if self.config.enable_reranking and self.reranking_manager and fallback_documents:
                            try:
                                reranked_fallback_documents = self.reranking_manager.rerank_documents(
                                    query=question,
                                    documents=fallback_documents
                                )
                                logger.debug(f"Re-ranked {len(fallback_documents)} documents for fallback")
                            except Exception as e:
                                logger.warning(f"Re-ranking failed for fallback, using original documents: {e}")
                                reranked_fallback_documents = fallback_documents
                        else:
                            reranked_fallback_documents = fallback_documents
                        
                        response = {
                            "answer": generation_result["replies"][0],
                            "source_documents": reranked_fallback_documents,
                            "meta": generation_result.get("meta", {}),
                            "provider_used": "anthropic",
                            "rag_mode": effective_rag_mode
                        }
                        
                        # Cache the fallback result if enabled
                        if self.config.cache_query_results and self.cache_manager and CACHE_AVAILABLE:
                            # Use semantic caching if we have the embedding
                            if query_embedding:
                                self.cache_manager.set_semantic_query_cache(
                                    question,
                                    query_embedding,
                                    response,
                                    filters=filters, 
                                    namespace=namespace,
                                    model=self.config.fallback_llm,
                                    extra_key=f"mode_{effective_rag_mode}"
                                )
                            else:
                                # Fallback to regular cache
                            self.cache_manager.set_query_cache(
                                question, 
                                response,
                                filters=filters, 
                                namespace=namespace,
                                    model=self.config.fallback_llm,
                                    extra_key=f"mode_{effective_rag_mode}"
                            )
                        
                        return response
                        
                    except Exception as fallback_error:
                        logger.error(f"Fallback LLM also failed: {fallback_error}")
                
                # Both failed
                return {
                    "answer": "I apologize, but I'm unable to process your query at the moment due to technical issues.",
                    "source_documents": [],
                    "meta": {"error": str(primary_error)},
                    "provider_used": "none",
                    "rag_mode": effective_rag_mode
                }
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": "I apologize, but an error occurred while processing your query.",
                "source_documents": [],
                "meta": {"error": str(e)},
                "provider_used": "none",
                "rag_mode": effective_rag_mode if 'effective_rag_mode' in locals() else self.config.rag_mode
            }
    
    def get_document_count(self) -> int:
        """Get total number of documents in the store"""
        try:
            return self.document_store.count_documents()
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def delete_documents(self, filters: Optional[Dict[str, Any]] = None) -> bool:
        """Delete documents from the store"""
        try:
            self.document_store.delete_documents(filters=filters)
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False


# Factory function for easy instantiation
def create_rag_pipeline(
    pinecone_index: str = "haystack-rag",
    embedding_model: str = "text-embedding-ada-002",
    primary_llm: str = "gpt-4o-mini",
    **kwargs
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline with custom settings
    
    Args:
        pinecone_index: Name of the Pinecone index
        embedding_model: Embedding model to use
        primary_llm: Primary LLM model
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured RAGPipeline instance
    """
    config = RAGConfig(
        pinecone_index=pinecone_index,
        embedding_model=embedding_model,
        primary_llm=primary_llm,
        **kwargs
    )
    
    return RAGPipeline(config)


# Example usage and testing
if __name__ == "__main__":
    # Example: Create and test the RAG pipeline
    import sys
    import tempfile
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create pipeline
        rag = create_rag_pipeline(
            pinecone_index="test-haystack-rag",
            primary_llm="gpt-4o-mini"
        )
        
        # Test with sample documents
        sample_docs = [
            Document(
                content="Haystack is an open-source framework for building search systems and LLM applications.",
                meta={"source": "haystack_docs", "type": "overview"}
            ),
            Document(
                content="Pinecone is a vector database designed for machine learning applications.",
                meta={"source": "pinecone_docs", "type": "overview"}
            ),
            Document(
                content="RAG (Retrieval-Augmented Generation) combines information retrieval with text generation.",
                meta={"source": "rag_docs", "type": "concept"}
            )
        ]
        
        # Index documents
        print("Indexing sample documents...")
        index_result = rag.index_documents(sample_docs)
        print(f"Indexing result: {index_result}")
        
        # Query the system
        print("\nQuerying the system...")
        query_result = rag.query("What is Haystack?")
        print(f"Answer: {query_result['answer']}")
        print(f"Provider used: {query_result['provider_used']}")
        print(f"Source documents: {len(query_result['source_documents'])}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        sys.exit(1) 