#!/usr/bin/env python3
"""
Multi-Model Support and Fallback Logic for RAG Pipeline

This module provides comprehensive model management capabilities including:
- Multiple LLM provider support (OpenAI, Anthropic)
- Intelligent fallback logic with health monitoring
- Advanced retry mechanisms with exponential backoff
- Per-provider rate limiting using token bucket algorithm
- Cost optimization and usage analytics
- Centralized configuration management

Author: AI Assistant
Date: 2025-05-27
"""

import asyncio
import time
import logging
import threading
import random
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import uuid

# Optional dependencies with graceful fallback
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)

class ProviderType(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class ModelCapability(Enum):
    """Model capability types."""
    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"
    FUNCTION_CALLING = "function_calling"

class ProviderHealth(Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"

class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    IMMEDIATE_FALLBACK = "immediate_fallback"
    NO_RETRY = "no_retry"

class Priority(Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ModelInfo:
    """Information about a specific model."""
    id: str
    provider: ProviderType
    display_name: str
    context_window: int
    input_cost_per_token: float  # Cost in USD per input token
    output_cost_per_token: float  # Cost in USD per output token
    capabilities: List[ModelCapability]
    max_tokens_per_minute: int = 10000
    is_available: bool = True
    description: str = ""

@dataclass
class ProviderConfig:
    """Configuration for a specific provider."""
    provider_type: ProviderType
    api_key: str
    base_url: Optional[str] = None
    rate_limit_tokens_per_minute: int = 10000
    rate_limit_requests_per_minute: int = 1000
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_retries: int = 3
    timeout_seconds: float = 30.0
    health_check_interval: float = 60.0

@dataclass
class FallbackChain:
    """Defines a fallback chain for model selection."""
    name: str
    primary_models: List[str]
    secondary_models: List[str]
    tertiary_models: List[str]
    preserve_capabilities: bool = True
    max_cost_multiplier: float = 2.0

@dataclass
class GenerationRequest:
    """Request for text generation."""
    prompt: str
    max_tokens: int = 1000
    temperature: float = 0.7
    model_preference: Optional[str] = None
    fallback_chain: str = "default"
    priority: Priority = Priority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenerationResponse:
    """Response from text generation."""
    text: str
    model_used: str
    provider_used: ProviderType
    tokens_used: int
    cost_usd: float
    latency_ms: float
    retries_attempted: int = 0
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProviderStats:
    """Statistics for a provider."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    average_latency_ms: float = 0.0
    last_error: Optional[str] = None
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    health_status: ProviderHealth = ProviderHealth.UNKNOWN

class TokenBucket:
    """Token bucket implementation for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int) -> bool:
        """Try to consume tokens. Returns True if successful."""
        with self._lock:
            now = time.time()
            # Refill tokens based on time passed
            time_passed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + time_passed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def tokens_available(self) -> int:
        """Get current number of available tokens."""
        with self._lock:
            now = time.time()
            time_passed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + time_passed * self.refill_rate)
            return int(self.tokens)

class RetryManager:
    """Manages retry logic with exponential backoff and jitter."""
    
    def __init__(self):
        self.base_delay = 1.0
        self.max_delay = 60.0
        self.jitter_factor = 0.25
    
    def calculate_delay(self, attempt: int, strategy: RetryStrategy) -> float:
        """Calculate delay for retry attempt."""
        if strategy == RetryStrategy.NO_RETRY:
            return 0.0
        elif strategy == RetryStrategy.IMMEDIATE_FALLBACK:
            return 0.1  # Minimal delay for immediate fallback
        else:  # EXPONENTIAL_BACKOFF
            delay = min(self.base_delay * (2 ** attempt), self.max_delay)
            jitter = delay * self.jitter_factor * (random.random() * 2 - 1)
            return max(0.1, delay + jitter)
    
    async def retry_with_backoff(
        self,
        func,
        max_retries: int,
        strategy: RetryStrategy,
        *args,
        **kwargs
    ):
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    delay = self.calculate_delay(attempt, strategy)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
        
        raise last_exception

class ModelRegistry:
    """Registry of available models and their capabilities."""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize with default model configurations."""
        # OpenAI Models
        if OPENAI_AVAILABLE:
            self.models.update({
                "gpt-4o": ModelInfo(
                    id="gpt-4o",
                    provider=ProviderType.OPENAI,
                    display_name="GPT-4o",
                    context_window=128000,
                    input_cost_per_token=0.005 / 1000,   # $5 per 1M input tokens
                    output_cost_per_token=0.015 / 1000,  # $15 per 1M output tokens
                    capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT_COMPLETION, 
                                ModelCapability.MULTIMODAL, ModelCapability.FUNCTION_CALLING],
                    max_tokens_per_minute=800000,
                    description="Most capable GPT-4 model with vision and function calling"
                ),
                "gpt-4o-mini": ModelInfo(
                    id="gpt-4o-mini",
                    provider=ProviderType.OPENAI,
                    display_name="GPT-4o-mini",
                    context_window=128000,
                    input_cost_per_token=0.00015 / 1000,  # $0.15 per 1M input tokens
                    output_cost_per_token=0.0006 / 1000,  # $0.60 per 1M output tokens
                    capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT_COMPLETION,
                                ModelCapability.FUNCTION_CALLING],
                    max_tokens_per_minute=2000000,
                    description="Fast and cost-effective GPT-4 model"
                ),
                "gpt-3.5-turbo": ModelInfo(
                    id="gpt-3.5-turbo",
                    provider=ProviderType.OPENAI,
                    display_name="GPT-3.5 Turbo",
                    context_window=16385,
                    input_cost_per_token=0.0005 / 1000,   # $0.50 per 1M input tokens
                    output_cost_per_token=0.0015 / 1000,  # $1.50 per 1M output tokens
                    capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT_COMPLETION,
                                ModelCapability.FUNCTION_CALLING],
                    max_tokens_per_minute=10000000,
                    description="Fast and economical model for simple tasks"
                )
            })
        
        # Anthropic Models
        if ANTHROPIC_AVAILABLE:
            self.models.update({
                "claude-3-5-sonnet-20241022": ModelInfo(
                    id="claude-3-5-sonnet-20241022",
                    provider=ProviderType.ANTHROPIC,
                    display_name="Claude 3.5 Sonnet",
                    context_window=200000,
                    input_cost_per_token=0.003 / 1000,    # $3 per 1M input tokens
                    output_cost_per_token=0.015 / 1000,   # $15 per 1M output tokens
                    capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT_COMPLETION,
                                ModelCapability.MULTIMODAL],
                    max_tokens_per_minute=400000,
                    description="Most capable Claude model with vision"
                ),
                "claude-3-5-haiku-20241022": ModelInfo(
                    id="claude-3-5-haiku-20241022",
                    provider=ProviderType.ANTHROPIC,
                    display_name="Claude 3.5 Haiku",
                    context_window=200000,
                    input_cost_per_token=0.0008 / 1000,   # $0.80 per 1M input tokens
                    output_cost_per_token=0.004 / 1000,   # $4 per 1M output tokens
                    capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT_COMPLETION],
                    max_tokens_per_minute=1000000,
                    description="Fast and cost-effective Claude model"
                ),
                "claude-3-opus-20240229": ModelInfo(
                    id="claude-3-opus-20240229",
                    provider=ProviderType.ANTHROPIC,
                    display_name="Claude 3 Opus",
                    context_window=200000,
                    input_cost_per_token=0.015 / 1000,    # $15 per 1M input tokens
                    output_cost_per_token=0.075 / 1000,   # $75 per 1M output tokens
                    capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT_COMPLETION,
                                ModelCapability.MULTIMODAL],
                    max_tokens_per_minute=200000,
                    description="Highest capability Claude model"
                )
            })
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID."""
        return self.models.get(model_id)
    
    def get_models_by_provider(self, provider: ProviderType) -> List[ModelInfo]:
        """Get all models for a specific provider."""
        return [model for model in self.models.values() if model.provider == provider]
    
    def get_models_by_capability(self, capability: ModelCapability) -> List[ModelInfo]:
        """Get all models with a specific capability."""
        return [model for model in self.models.values() if capability in model.capabilities]
    
    def register_model(self, model: ModelInfo):
        """Register a new model."""
        self.models[model.id] = model

class ProviderManager:
    """Manages LLM provider instances and health monitoring."""
    
    def __init__(self):
        self.providers: Dict[ProviderType, Any] = {}
        self.configs: Dict[ProviderType, ProviderConfig] = {}
        self.stats: Dict[ProviderType, ProviderStats] = defaultdict(ProviderStats)
        self.rate_limiters: Dict[ProviderType, TokenBucket] = {}
        self.health_monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
    
    def register_provider(self, config: ProviderConfig):
        """Register a provider with configuration."""
        self.configs[config.provider_type] = config
        
        # Initialize rate limiter
        self.rate_limiters[config.provider_type] = TokenBucket(
            capacity=config.rate_limit_tokens_per_minute,
            refill_rate=config.rate_limit_tokens_per_minute / 60.0  # per second
        )
        
        # Initialize provider client
        try:
            if config.provider_type == ProviderType.OPENAI and OPENAI_AVAILABLE:
                self.providers[config.provider_type] = openai.AsyncOpenAI(
                    api_key=config.api_key,
                    base_url=config.base_url,
                    timeout=config.timeout_seconds
                )
                self.stats[config.provider_type].health_status = ProviderHealth.HEALTHY
                logger.info(f"OpenAI provider registered successfully")
                
            elif config.provider_type == ProviderType.ANTHROPIC and ANTHROPIC_AVAILABLE:
                self.providers[config.provider_type] = anthropic.AsyncAnthropic(
                    api_key=config.api_key,
                    base_url=config.base_url,
                    timeout=config.timeout_seconds
                )
                self.stats[config.provider_type].health_status = ProviderHealth.HEALTHY
                logger.info(f"Anthropic provider registered successfully")
                
        except Exception as e:
            logger.error(f"Failed to register {config.provider_type.value} provider: {e}")
            self.stats[config.provider_type].health_status = ProviderHealth.DOWN
            self.stats[config.provider_type].last_error = str(e)
    
    def get_provider_health(self, provider: ProviderType) -> ProviderHealth:
        """Get current health status of a provider."""
        return self.stats[provider].health_status
    
    def update_provider_stats(
        self,
        provider: ProviderType,
        success: bool,
        tokens: int = 0,
        cost: float = 0.0,
        latency_ms: float = 0.0,
        error: Optional[str] = None
    ):
        """Update provider statistics."""
        with self._lock:
            stats = self.stats[provider]
            stats.total_requests += 1
            
            if success:
                stats.successful_requests += 1
                stats.total_tokens += tokens
                stats.total_cost_usd += cost
                stats.last_success_time = time.time()
                
                # Update average latency
                if stats.successful_requests > 1:
                    stats.average_latency_ms = (
                        (stats.average_latency_ms * (stats.successful_requests - 1) + latency_ms) /
                        stats.successful_requests
                    )
                else:
                    stats.average_latency_ms = latency_ms
                
                # Update health status based on success rate
                success_rate = stats.successful_requests / stats.total_requests
                if success_rate >= 0.95:
                    stats.health_status = ProviderHealth.HEALTHY
                elif success_rate >= 0.8:
                    stats.health_status = ProviderHealth.DEGRADED
                else:
                    stats.health_status = ProviderHealth.DOWN
            else:
                stats.failed_requests += 1
                stats.last_failure_time = time.time()
                stats.last_error = error
                
                # Update health status
                failure_rate = stats.failed_requests / stats.total_requests
                if failure_rate >= 0.5:
                    stats.health_status = ProviderHealth.DOWN
                elif failure_rate >= 0.2:
                    stats.health_status = ProviderHealth.DEGRADED

class CostOptimizer:
    """Optimizes model selection based on cost and performance."""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.usage_history: List[Dict[str, Any]] = []
        self.cost_thresholds = {
            Priority.LOW: 0.01,      # $0.01 max per request
            Priority.NORMAL: 0.05,   # $0.05 max per request
            Priority.HIGH: 0.20,     # $0.20 max per request
            Priority.CRITICAL: 1.0   # $1.00 max per request
        }
    
    def estimate_cost(self, model_id: str, prompt_tokens: int, max_tokens: int) -> float:
        """Estimate cost for a request."""
        model = self.model_registry.get_model(model_id)
        if not model:
            return float('inf')
        
        input_cost = prompt_tokens * model.input_cost_per_token
        output_cost = max_tokens * model.output_cost_per_token
        return input_cost + output_cost
    
    def select_optimal_model(
        self,
        available_models: List[str],
        prompt_tokens: int,
        max_tokens: int,
        priority: Priority,
        required_capabilities: List[ModelCapability] = None
    ) -> Optional[str]:
        """Select the most cost-effective model for the request."""
        required_capabilities = required_capabilities or [ModelCapability.TEXT_GENERATION]
        max_cost = self.cost_thresholds.get(priority, float('inf'))
        
        # Filter models by capabilities and availability
        eligible_models = []
        for model_id in available_models:
            model = self.model_registry.get_model(model_id)
            if (model and model.is_available and 
                all(cap in model.capabilities for cap in required_capabilities)):
                cost = self.estimate_cost(model_id, prompt_tokens, max_tokens)
                if cost <= max_cost:
                    eligible_models.append((model_id, cost, model))
        
        if not eligible_models:
            return None
        
        # Sort by cost and select the cheapest that meets requirements
        eligible_models.sort(key=lambda x: x[1])
        return eligible_models[0][0]
    
    def record_usage(self, model_id: str, tokens_used: int, cost: float, latency_ms: float):
        """Record usage for analytics."""
        self.usage_history.append({
            'timestamp': time.time(),
            'model_id': model_id,
            'tokens_used': tokens_used,
            'cost': cost,
            'latency_ms': latency_ms
        })
        
        # Keep only last 1000 entries
        if len(self.usage_history) > 1000:
            self.usage_history = self.usage_history[-1000:]

class ModelManager:
    """Central manager for multi-model support with fallback logic."""
    
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.provider_manager = ProviderManager()
        self.cost_optimizer = CostOptimizer(self.model_registry)
        self.retry_manager = RetryManager()
        self.fallback_chains: Dict[str, FallbackChain] = {}
        self._initialize_default_fallback_chains()
        logger.info("ModelManager initialized successfully")
    
    def _initialize_default_fallback_chains(self):
        """Initialize default fallback chains."""
        self.fallback_chains = {
            "default": FallbackChain(
                name="default",
                primary_models=["gpt-4o-mini", "claude-3-5-haiku-20241022"],
                secondary_models=["gpt-3.5-turbo", "claude-3-5-sonnet-20241022"],
                tertiary_models=["gpt-4o", "claude-3-opus-20240229"]
            ),
            "high_capability": FallbackChain(
                name="high_capability",
                primary_models=["gpt-4o", "claude-3-5-sonnet-20241022"],
                secondary_models=["claude-3-opus-20240229", "gpt-4o-mini"],
                tertiary_models=["gpt-3.5-turbo", "claude-3-5-haiku-20241022"]
            ),
            "cost_optimized": FallbackChain(
                name="cost_optimized",
                primary_models=["gpt-3.5-turbo", "claude-3-5-haiku-20241022"],
                secondary_models=["gpt-4o-mini"],
                tertiary_models=["claude-3-5-sonnet-20241022", "gpt-4o"]
            )
        }
    
    def register_provider(self, config: ProviderConfig):
        """Register a provider."""
        self.provider_manager.register_provider(config)
    
    def _get_available_models_from_chain(self, chain_name: str) -> List[str]:
        """Get available models from a fallback chain."""
        chain = self.fallback_chains.get(chain_name, self.fallback_chains["default"])
        all_models = chain.primary_models + chain.secondary_models + chain.tertiary_models
        
        # Filter by provider health and model availability
        available_models = []
        for model_id in all_models:
            model = self.model_registry.get_model(model_id)
            if model and model.is_available:
                provider_health = self.provider_manager.get_provider_health(model.provider)
                if provider_health in [ProviderHealth.HEALTHY, ProviderHealth.DEGRADED]:
                    available_models.append(model_id)
        
        return available_models
    
    def _select_model_with_fallback(self, request: GenerationRequest) -> Tuple[str, bool]:
        """Select model with fallback logic."""
        # If specific model requested, try it first
        if request.model_preference:
            model = self.model_registry.get_model(request.model_preference)
            if model and model.is_available:
                provider_health = self.provider_manager.get_provider_health(model.provider)
                if provider_health != ProviderHealth.DOWN:
                    return request.model_preference, False
        
        # Get available models from fallback chain
        available_models = self._get_available_models_from_chain(request.fallback_chain)
        
        if not available_models:
            raise Exception("No available models found in any fallback chain")
        
        # Use cost optimizer to select best model
        prompt_tokens = len(request.prompt.split()) * 1.3  # Rough token estimation
        selected_model = self.cost_optimizer.select_optimal_model(
            available_models,
            int(prompt_tokens),
            request.max_tokens,
            request.priority
        )
        
        if not selected_model:
            # Fallback to first available model if cost optimizer fails
            selected_model = available_models[0]
        
        fallback_used = selected_model != request.model_preference
        return selected_model, fallback_used
    
    async def _generate_with_provider(
        self,
        provider_type: ProviderType,
        model_id: str,
        request: GenerationRequest
    ) -> GenerationResponse:
        """Generate text using a specific provider."""
        provider = self.provider_manager.providers.get(provider_type)
        if not provider:
            raise Exception(f"Provider {provider_type.value} not available")
        
        # Check rate limiting
        rate_limiter = self.provider_manager.rate_limiters.get(provider_type)
        if rate_limiter and not rate_limiter.consume(request.max_tokens):
            raise Exception(f"Rate limit exceeded for {provider_type.value}")
        
        start_time = time.time()
        
        try:
            if provider_type == ProviderType.OPENAI:
                response = await provider.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": request.prompt}],
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                
                generated_text = response.choices[0].message.content
                tokens_used = response.usage.total_tokens
                
            elif provider_type == ProviderType.ANTHROPIC:
                response = await provider.messages.create(
                    model=model_id,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    messages=[{"role": "user", "content": request.prompt}]
                )
                
                generated_text = response.content[0].text
                tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            else:
                raise Exception(f"Unsupported provider: {provider_type}")
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate cost
            model_info = self.model_registry.get_model(model_id)
            cost = 0.0
            if model_info:
                if provider_type == ProviderType.OPENAI and hasattr(response, 'usage'):
                    cost = (response.usage.prompt_tokens * model_info.input_cost_per_token +
                           response.usage.completion_tokens * model_info.output_cost_per_token)
                elif provider_type == ProviderType.ANTHROPIC and hasattr(response, 'usage'):
                    cost = (response.usage.input_tokens * model_info.input_cost_per_token +
                           response.usage.output_tokens * model_info.output_cost_per_token)
            
            # Update provider stats
            self.provider_manager.update_provider_stats(
                provider_type, True, tokens_used, cost, latency_ms
            )
            
            # Record usage for analytics
            self.cost_optimizer.record_usage(model_id, tokens_used, cost, latency_ms)
            
            return GenerationResponse(
                text=generated_text,
                model_used=model_id,
                provider_used=provider_type,
                tokens_used=tokens_used,
                cost_usd=cost,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.provider_manager.update_provider_stats(
                provider_type, False, latency_ms=latency_ms, error=str(e)
            )
            raise e
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text with automatic fallback and retry logic."""
        # Select model with fallback
        selected_model, fallback_used = self._select_model_with_fallback(request)
        model_info = self.model_registry.get_model(selected_model)
        
        if not model_info:
            raise Exception(f"Model {selected_model} not found in registry")
        
        # Get provider configuration
        provider_config = self.provider_manager.configs.get(model_info.provider)
        if not provider_config:
            raise Exception(f"Provider {model_info.provider.value} not configured")
        
        # Attempt generation with retry logic
        try:
            response = await self.retry_manager.retry_with_backoff(
                self._generate_with_provider,
                provider_config.max_retries,
                provider_config.retry_strategy,
                model_info.provider,
                selected_model,
                request
            )
            
            response.fallback_used = fallback_used
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate with model {selected_model}: {e}")
            
            # Try fallback models if primary failed
            if not fallback_used:
                available_models = self._get_available_models_from_chain(request.fallback_chain)
                for fallback_model in available_models[1:]:  # Skip first (already tried)
                    try:
                        fallback_model_info = self.model_registry.get_model(fallback_model)
                        if fallback_model_info:
                            response = await self._generate_with_provider(
                                fallback_model_info.provider,
                                fallback_model,
                                request
                            )
                            response.fallback_used = True
                            response.retries_attempted = provider_config.max_retries + 1
                            return response
                    except Exception as fallback_error:
                        logger.warning(f"Fallback model {fallback_model} also failed: {fallback_error}")
                        continue
            
            raise Exception(f"All models failed. Last error: {e}")
    
    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all providers."""
        return {
            provider.value: asdict(stats)
            for provider, stats in self.provider_manager.stats.items()
        }
    
    def get_cost_analytics(self) -> Dict[str, Any]:
        """Get cost analytics and usage statistics."""
        if not self.cost_optimizer.usage_history:
            return {"total_cost": 0.0, "total_requests": 0, "models_used": {}}
        
        total_cost = sum(entry['cost'] for entry in self.cost_optimizer.usage_history)
        total_requests = len(self.cost_optimizer.usage_history)
        
        model_usage = defaultdict(list)
        for entry in self.cost_optimizer.usage_history:
            model_usage[entry['model_id']].append(entry)
        
        models_stats = {}
        for model_id, entries in model_usage.items():
            models_stats[model_id] = {
                "requests": len(entries),
                "total_cost": sum(e['cost'] for e in entries),
                "avg_latency_ms": sum(e['latency_ms'] for e in entries) / len(entries),
                "total_tokens": sum(e['tokens_used'] for e in entries)
            }
        
        return {
            "total_cost": total_cost,
            "total_requests": total_requests,
            "average_cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "models_used": models_stats
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        provider_health = {}
        for provider_type, stats in self.provider_manager.stats.items():
            provider_health[provider_type.value] = {
                "status": stats.health_status.value,
                "success_rate": (stats.successful_requests / max(stats.total_requests, 1)),
                "avg_latency_ms": stats.average_latency_ms,
                "last_error": stats.last_error
            }
        
        # Determine overall health
        healthy_providers = sum(1 for stats in self.provider_manager.stats.values() 
                              if stats.health_status == ProviderHealth.HEALTHY)
        total_providers = len(self.provider_manager.stats)
        
        if healthy_providers == total_providers:
            overall_status = "healthy"
        elif healthy_providers > 0:
            overall_status = "degraded"
        else:
            overall_status = "down"
        
        return {
            "overall_status": overall_status,
            "healthy_providers": healthy_providers,
            "total_providers": total_providers,
            "providers": provider_health
        }

# Singleton instance
_model_manager_instance: Optional[ModelManager] = None
_model_manager_lock = threading.Lock()

def get_model_manager() -> ModelManager:
    """Get singleton instance of ModelManager."""
    global _model_manager_instance
    
    if _model_manager_instance is None:
        with _model_manager_lock:
            if _model_manager_instance is None:
                _model_manager_instance = ModelManager()
    
    return _model_manager_instance

def initialize_model_manager(provider_configs: List[ProviderConfig]) -> ModelManager:
    """Initialize the model manager with provider configurations."""
    manager = get_model_manager()
    
    for config in provider_configs:
        manager.register_provider(config)
    
    logger.info(f"Model manager initialized with {len(provider_configs)} providers")
    return manager 