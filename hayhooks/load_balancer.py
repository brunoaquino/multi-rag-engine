#!/usr/bin/env python3
"""
Load Balancer for RAG Pipeline Optimization

This module provides load balancing capabilities including:
- Query queue management for handling concurrent requests
- Resource pool for efficient resource allocation
- Circuit breaker pattern for system protection
- Load distribution strategies
- Performance monitoring and metrics

Author: AI Assistant
Date: 2025-05-27
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
import queue

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class QueryPriority(Enum):
    """Query priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class QueryRequest:
    """Represents a query request in the system."""
    id: str
    query: str
    priority: QueryPriority = QueryPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    timeout: float = 30.0  # seconds
    callback: Optional[Callable] = None

@dataclass
class ResourceWorker:
    """Represents a worker resource."""
    id: str
    name: str
    weight: int = 1
    max_concurrent: int = 5
    current_load: int = 0
    total_processed: int = 0
    total_errors: int = 0
    last_used: float = 0.0
    is_healthy: bool = True
    processor_func: Optional[Callable] = None

@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer."""
    max_queue_size: int = 1000
    max_workers: int = 10
    worker_timeout: float = 30.0
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS
    
    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    
    # Queue settings
    enable_priority_queue: bool = True
    max_wait_time: float = 60.0
    
    # Health check settings
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0

class CircuitBreaker:
    """Circuit breaker implementation for system protection."""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time > self.config.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    return True
                return False
            elif self.state == CircuitBreakerState.HALF_OPEN:
                return self.half_open_calls < self.config.half_open_max_calls
            
            return False
    
    def record_success(self):
        """Record successful execution."""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.half_open_calls += 1
                if self.half_open_calls >= self.config.half_open_max_calls:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker transitioning to CLOSED")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed execution."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning("Circuit breaker transitioning to OPEN from HALF_OPEN")
            elif (self.state == CircuitBreakerState.CLOSED and 
                  self.failure_count >= self.config.failure_threshold):
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker OPEN: {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        with self._lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "last_failure_time": self.last_failure_time,
                "half_open_calls": self.half_open_calls
            }

class QueryQueue:
    """Priority queue for managing query requests."""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self._queues = {
            QueryPriority.CRITICAL: queue.PriorityQueue(),
            QueryPriority.HIGH: queue.PriorityQueue(),
            QueryPriority.NORMAL: queue.PriorityQueue(),
            QueryPriority.LOW: queue.PriorityQueue()
        }
        self._size = 0
        self._lock = threading.Lock()
        self._metrics = {
            "total_enqueued": 0,
            "total_dequeued": 0,
            "total_timeouts": 0,
            "current_size": 0,
            "priority_distribution": defaultdict(int)
        }
    
    def enqueue(self, request: QueryRequest) -> bool:
        """Add request to queue."""
        with self._lock:
            if self._size >= self.config.max_queue_size:
                logger.warning(f"Queue full, rejecting request {request.id}")
                return False
            
            # Use negative timestamp for FIFO within same priority
            priority_value = (-request.priority.value, request.timestamp)
            self._queues[request.priority].put((priority_value, request))
            self._size += 1
            
            # Update metrics
            self._metrics["total_enqueued"] += 1
            self._metrics["current_size"] = self._size
            self._metrics["priority_distribution"][request.priority.value] += 1
            
            return True
    
    def dequeue(self, timeout: float = 1.0) -> Optional[QueryRequest]:
        """Get next request from queue."""
        # Check queues in priority order
        for priority in [QueryPriority.CRITICAL, QueryPriority.HIGH, 
                        QueryPriority.NORMAL, QueryPriority.LOW]:
            try:
                priority_value, request = self._queues[priority].get(timeout=timeout)
                
                with self._lock:
                    self._size -= 1
                    self._metrics["total_dequeued"] += 1
                    self._metrics["current_size"] = self._size
                
                # Check if request has timed out
                if time.time() - request.timestamp > request.timeout:
                    with self._lock:
                        self._metrics["total_timeouts"] += 1
                    logger.warning(f"Request {request.id} timed out")
                    continue
                
                return request
                
            except queue.Empty:
                continue
        
        return None
    
    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return self._size
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics."""
        with self._lock:
            return dict(self._metrics)

class ResourcePool:
    """Pool of worker resources for processing queries."""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.workers: Dict[str, ResourceWorker] = {}
        self._lock = threading.Lock()
        self._strategy_state = {"current_index": 0}  # For round robin
    
    def add_worker(self, worker: ResourceWorker):
        """Add worker to pool."""
        with self._lock:
            self.workers[worker.id] = worker
            logger.info(f"Added worker {worker.id} to pool")
    
    def remove_worker(self, worker_id: str):
        """Remove worker from pool."""
        with self._lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                logger.info(f"Removed worker {worker_id} from pool")
    
    def get_available_worker(self) -> Optional[ResourceWorker]:
        """Get next available worker based on strategy."""
        with self._lock:
            available_workers = [
                worker for worker in self.workers.values()
                if worker.is_healthy and worker.current_load < worker.max_concurrent
            ]
            
            if not available_workers:
                return None
            
            if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(available_workers)
            elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(available_workers)
            elif self.config.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(available_workers)
            else:  # RANDOM
                import random
                return random.choice(available_workers)
    
    def _round_robin_selection(self, workers: List[ResourceWorker]) -> ResourceWorker:
        """Round robin worker selection."""
        if not workers:
            return None
        
        index = self._strategy_state["current_index"] % len(workers)
        self._strategy_state["current_index"] = (index + 1) % len(workers)
        return workers[index]
    
    def _least_connections_selection(self, workers: List[ResourceWorker]) -> ResourceWorker:
        """Select worker with least connections."""
        return min(workers, key=lambda w: w.current_load)
    
    def _weighted_round_robin_selection(self, workers: List[ResourceWorker]) -> ResourceWorker:
        """Weighted round robin selection."""
        # Simple weighted selection based on worker weight
        total_weight = sum(w.weight for w in workers)
        if total_weight == 0:
            return workers[0] if workers else None
        
        # Use current time as seed for selection
        import random
        random.seed(int(time.time() * 1000) % 1000)
        target = random.randint(1, total_weight)
        
        current_weight = 0
        for worker in workers:
            current_weight += worker.weight
            if current_weight >= target:
                return worker
        
        return workers[-1]  # Fallback
    
    def acquire_worker(self, worker: ResourceWorker):
        """Mark worker as busy."""
        with self._lock:
            worker.current_load += 1
            worker.last_used = time.time()
    
    def release_worker(self, worker: ResourceWorker, success: bool = True):
        """Mark worker as available."""
        with self._lock:
            worker.current_load = max(0, worker.current_load - 1)
            worker.total_processed += 1
            if not success:
                worker.total_errors += 1
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        with self._lock:
            stats = {}
            for worker_id, worker in self.workers.items():
                stats[worker_id] = {
                    "name": worker.name,
                    "current_load": worker.current_load,
                    "max_concurrent": worker.max_concurrent,
                    "total_processed": worker.total_processed,
                    "total_errors": worker.total_errors,
                    "error_rate": worker.total_errors / max(worker.total_processed, 1),
                    "is_healthy": worker.is_healthy,
                    "last_used": worker.last_used
                }
            return stats

@dataclass
class LoadBalancerMetrics:
    """Metrics for load balancer performance."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    average_response_time: float = 0.0
    current_queue_size: int = 0
    active_workers: int = 0
    circuit_breaker_trips: int = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get formatted statistics."""
        success_rate = self.successful_requests / max(self.total_requests, 1)
        failure_rate = self.failed_requests / max(self.total_requests, 1)
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rejected_requests": self.rejected_requests,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "average_response_time": self.average_response_time,
            "current_queue_size": self.current_queue_size,
            "active_workers": self.active_workers,
            "circuit_breaker_trips": self.circuit_breaker_trips
        }

class LoadBalancer:
    """Main load balancer for distributing query processing."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        
        # Core components
        self.query_queue = QueryQueue(self.config)
        self.resource_pool = ResourcePool(self.config)
        self.circuit_breaker = CircuitBreaker(self.config)
        
        # Metrics and monitoring
        self.metrics = LoadBalancerMetrics()
        self._metrics_lock = threading.Lock()
        
        # Processing
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._running = False
        self._processor_thread = None
        
        logger.info("Load Balancer initialized")
    
    def start(self):
        """Start the load balancer."""
        if self._running:
            return
        
        self._running = True
        self._processor_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._processor_thread.start()
        logger.info("Load Balancer started")
    
    def stop(self):
        """Stop the load balancer."""
        self._running = False
        if self._processor_thread:
            self._processor_thread.join(timeout=5.0)
        self._executor.shutdown(wait=True)
        logger.info("Load Balancer stopped")
    
    def submit_query(self, request: QueryRequest) -> bool:
        """Submit query for processing."""
        if not self.circuit_breaker.can_execute():
            with self._metrics_lock:
                self.metrics.rejected_requests += 1
            logger.warning(f"Request {request.id} rejected by circuit breaker")
            return False
        
        success = self.query_queue.enqueue(request)
        if not success:
            with self._metrics_lock:
                self.metrics.rejected_requests += 1
        
        return success
    
    def _process_queue(self):
        """Main processing loop."""
        while self._running:
            try:
                # Get next request
                request = self.query_queue.dequeue(timeout=1.0)
                if not request:
                    continue
                
                # Get available worker
                worker = self.resource_pool.get_available_worker()
                if not worker:
                    # No workers available, put request back
                    self.query_queue.enqueue(request)
                    time.sleep(0.1)
                    continue
                
                # Process request asynchronously
                future = self._executor.submit(self._process_request, request, worker)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(1.0)
    
    def _process_request(self, request: QueryRequest, worker: ResourceWorker):
        """Process individual request."""
        start_time = time.time()
        success = False
        
        try:
            # Acquire worker
            self.resource_pool.acquire_worker(worker)
            
            # Process request
            if worker.processor_func:
                result = worker.processor_func(request.query)
                success = True
                
                # Execute callback if provided
                if request.callback:
                    request.callback(result)
            else:
                logger.warning(f"Worker {worker.id} has no processor function")
            
            # Record success
            self.circuit_breaker.record_success()
            
        except Exception as e:
            logger.error(f"Error processing request {request.id}: {e}")
            self.circuit_breaker.record_failure()
            
            # Execute error callback if provided
            if request.callback:
                try:
                    request.callback(None, error=str(e))
                except Exception as callback_error:
                    logger.error(f"Error in callback: {callback_error}")
        
        finally:
            # Release worker
            self.resource_pool.release_worker(worker, success)
            
            # Update metrics
            processing_time = time.time() - start_time
            with self._metrics_lock:
                self.metrics.total_requests += 1
                if success:
                    self.metrics.successful_requests += 1
                else:
                    self.metrics.failed_requests += 1
                
                # Update average response time
                total_time = (self.metrics.average_response_time * 
                             (self.metrics.total_requests - 1) + processing_time)
                self.metrics.average_response_time = total_time / self.metrics.total_requests
    
    def add_worker(self, worker_id: str, name: str, processor_func: Callable, 
                   weight: int = 1, max_concurrent: int = 5):
        """Add worker to the pool."""
        worker = ResourceWorker(
            id=worker_id,
            name=name,
            weight=weight,
            max_concurrent=max_concurrent,
            processor_func=processor_func
        )
        self.resource_pool.add_worker(worker)
    
    def remove_worker(self, worker_id: str):
        """Remove worker from the pool."""
        self.resource_pool.remove_worker(worker_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics."""
        with self._metrics_lock:
            stats = self.metrics.get_stats()
        
        # Add queue metrics
        stats["queue_metrics"] = self.query_queue.get_metrics()
        
        # Add worker stats
        stats["worker_stats"] = self.resource_pool.get_worker_stats()
        
        # Add circuit breaker state
        stats["circuit_breaker"] = self.circuit_breaker.get_state()
        
        # Update current values
        stats["current_queue_size"] = self.query_queue.size()
        stats["active_workers"] = len([w for w in self.resource_pool.workers.values() 
                                      if w.current_load > 0])
        
        return stats
    
    def reset_stats(self):
        """Reset load balancer statistics."""
        with self._metrics_lock:
            self.metrics = LoadBalancerMetrics()

# Singleton instance
_load_balancer_instance = None
_load_balancer_lock = threading.Lock()

def get_load_balancer(config: Optional[LoadBalancerConfig] = None) -> LoadBalancer:
    """Get singleton load balancer instance."""
    global _load_balancer_instance
    
    with _load_balancer_lock:
        if _load_balancer_instance is None:
            _load_balancer_instance = LoadBalancer(config)
        return _load_balancer_instance

def reset_load_balancer():
    """Reset singleton load balancer instance."""
    global _load_balancer_instance
    
    with _load_balancer_lock:
        if _load_balancer_instance:
            _load_balancer_instance.stop()
        _load_balancer_instance = None

if __name__ == "__main__":
    # Test the load balancer
    import random
    
    def mock_processor(query: str) -> str:
        """Mock query processor."""
        time.sleep(random.uniform(0.1, 0.5))  # Simulate processing time
        return f"Processed: {query}"
    
    # Create load balancer
    config = LoadBalancerConfig(max_workers=3, max_queue_size=10)
    lb = LoadBalancer(config)
    
    # Add workers
    for i in range(3):
        lb.add_worker(f"worker_{i}", f"Worker {i}", mock_processor)
    
    # Start load balancer
    lb.start()
    
    # Submit test requests
    def result_callback(result, error=None):
        if error:
            print(f"Error: {error}")
        else:
            print(f"Result: {result}")
    
    for i in range(10):
        request = QueryRequest(
            id=str(uuid.uuid4()),
            query=f"Test query {i}",
            priority=QueryPriority.NORMAL,
            callback=result_callback
        )
        success = lb.submit_query(request)
        print(f"Submitted request {i}: {success}")
    
    # Wait for processing
    time.sleep(3)
    
    # Print stats
    print(f"\nLoad Balancer Stats:")
    stats = lb.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Stop load balancer
    lb.stop() 