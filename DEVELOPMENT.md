# 🛠️ Development Guide

Guia completo para desenvolvimento do Haystack RAG System.

## 📋 Índice

- [Setup Ambiente de Desenvolvimento](#setup-ambiente-de-desenvolvimento)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Desenvolvimento Local](#desenvolvimento-local)
- [Testing](#testing)
- [Debugging](#debugging)
- [Contributing](#contributing)
- [Best Practices](#best-practices)
- [Performance Guidelines](#performance-guidelines)

---

## 🚀 Setup Ambiente de Desenvolvimento

### Pré-requisitos

- **Python 3.11+**
- **Docker 20.10+**
- **Docker Compose 2.0+**
- **Git**
- **VS Code** (recomendado) ou seu editor preferido

### 1. Clone e Setup

```bash
# Clone o repositório
git clone <repository-url>
cd haystack-rag-system

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências de desenvolvimento
pip install -r requirements-dev.txt
```

### 2. Configuração do Ambiente

```bash
# Copiar configuração de desenvolvimento
cp local.env.example .env

# Editar com credenciais de desenvolvimento
nano .env
```

### 3. Setup de APIs de Desenvolvimento

#### OpenAI (Obrigatório)

```bash
# Para desenvolvimento, use uma chave com limite baixo
OPENAI_API_KEY=sk-proj-...
OPENAI_CHAT_MODEL=gpt-4o-mini  # Modelo mais barato para dev
```

#### Pinecone (Obrigatório)

```bash
# Use namespace separado para desenvolvimento
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=haystack-dev
PINECONE_NAMESPACE=dev-docs
```

### 4. Iniciar Ambiente

```bash
# Build e start serviços
docker-compose up -d

# Verificar se tudo está funcionando
curl http://localhost:8000/health
```

---

## 📁 Estrutura do Projeto

```
haystack-rag-system/
├── 📄 README.md                    # Documentação principal
├── 📄 DEVELOPMENT.md               # Este arquivo
├── 📄 docker-compose.yml           # Configuração Docker
├── 📄 local.env.example            # Template de variáveis
├── 📄 .env                         # Variáveis de ambiente (git ignored)
├── 📄 requirements.txt             # Dependências Python
├── 📄 requirements-dev.txt         # Dependências de desenvolvimento
│
├── 📂 hayhooks/                    # Backend principal
│   ├── 📄 Dockerfile               # Container do backend
│   ├── 📄 direct_api.py            # FastAPI application
│   ├── 📄 cache_manager.py         # Gerenciamento de cache Redis
│   ├── 📄 cached_embedder.py       # Embeddings com cache
│   ├── 📂 pipelines/               # Pipelines Haystack
│   │   ├── 📄 rag_pipeline.py      # Pipeline RAG principal
│   │   ├── 📄 chat_pipeline.py     # Pipeline de chat simples
│   │   └── 📂 components/          # Componentes customizados
│   ├── 📂 utils/                   # Utilitários
│   └── 📂 tests/                   # Testes unitários
│
├── 📂 open-webui/                  # Frontend (OpenWebUI)
│   └── 📄 Dockerfile               # Container do frontend
│
├── 📂 data/                        # Dados locais
│   ├── 📂 uploads/                 # Arquivos enviados
│   ├── 📂 processed/               # Documentos processados
│   └── 📂 cache/                   # Cache local
│
├── 📂 docs/                        # Documentação
│   ├── 📄 API.md                   # Documentação da API
│   ├── 📄 DEPLOYMENT.md            # Guia de deployment
│   ├── 📄 ARCHITECTURE.md          # Arquitetura do sistema
│   ├── 📄 CONFIGURATION.md         # Guia de configuração
│   └── 📄 TROUBLESHOOTING.md       # Solução de problemas
│
├── 📂 scripts/                     # Scripts de automação
│   ├── 📄 setup.sh                 # Script de setup
│   ├── 📄 deploy.sh                # Script de deploy
│   └── 📄 test.sh                  # Script de testes
│
└── 📂 nginx/                       # Configuração Nginx (produção)
    └── 📄 nginx.conf               # Configuração do proxy reverso
```

---

## 💻 Desenvolvimento Local

### Workflow de Desenvolvimento

#### 1. Desenvolvimento com Hot Reload

```bash
# Para desenvolvimento da API, use reload automático
cd hayhooks
uvicorn direct_api:app --reload --host 0.0.0.0 --port 8000

# Em outro terminal, mantenha outros serviços rodando
docker-compose up redis -d
```

#### 2. Modificações no Pipeline

```python
# hayhooks/pipelines/rag_pipeline.py
from haystack import Pipeline, component

@component
class CustomRAGComponent:
    """Exemplo de componente customizado"""

    def __init__(self, custom_param: str = "default"):
        self.custom_param = custom_param

    @component.output_types(result=str)
    def run(self, query: str) -> dict:
        # Sua lógica customizada aqui
        processed_query = f"{self.custom_param}: {query}"
        return {"result": processed_query}

# Integrar no pipeline
def create_custom_rag_pipeline():
    pipeline = Pipeline()

    # Adicionar componentes
    pipeline.add_component("custom", CustomRAGComponent("dev"))
    pipeline.add_component("embedder", cached_embedder)
    pipeline.add_component("retriever", pinecone_retriever)
    pipeline.add_component("generator", openai_generator)

    # Conectar componentes
    pipeline.connect("custom", "embedder")
    pipeline.connect("embedder", "retriever")
    pipeline.connect("retriever", "generator")

    return pipeline
```

#### 3. Adicionando Novos Endpoints

```python
# hayhooks/direct_api.py
from fastapi import APIRouter

# Criar router para organizar endpoints
dev_router = APIRouter(prefix="/dev", tags=["development"])

@dev_router.post("/test-embedding")
async def test_embedding(text: str):
    """Endpoint para testar embeddings durante desenvolvimento"""
    try:
        embedding = cached_embedder.run(text)
        return {
            "text": text,
            "embedding_size": len(embedding["embedding"]),
            "first_values": embedding["embedding"][:5]
        }
    except Exception as e:
        return {"error": str(e)}

@dev_router.get("/cache-info")
async def cache_info():
    """Informações detalhadas do cache para debugging"""
    try:
        info = redis_client.info()
        return {
            "memory_usage": info.get("used_memory_human"),
            "total_keys": redis_client.dbsize(),
            "cache_hits": info.get("keyspace_hits", 0),
            "cache_misses": info.get("keyspace_misses", 0)
        }
    except Exception as e:
        return {"error": str(e)}

# Adicionar ao app principal
app.include_router(dev_router)
```

### Configuração do VS Code

#### `.vscode/settings.json`

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/venv": true
  },
  "docker.defaultRegistryPath": "localhost:5000"
}
```

#### `.vscode/launch.json`

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "FastAPI Debug",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/venv/bin/uvicorn",
      "args": [
        "hayhooks.direct_api:app",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        "8000"
      ],
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "envFile": "${workspaceFolder}/.env"
    }
  ]
}
```

---

## 🧪 Testing

### Setup de Testes

```bash
# Instalar dependências de teste
pip install pytest pytest-asyncio pytest-cov httpx

# Executar testes
pytest hayhooks/tests/ -v

# Com coverage
pytest hayhooks/tests/ --cov=hayhooks --cov-report=html
```

### Estrutura de Testes

```python
# hayhooks/tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from hayhooks.direct_api import app

client = TestClient(app)

def test_health_endpoint():
    """Teste básico do endpoint de health"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data

@pytest.mark.asyncio
async def test_rag_query():
    """Teste do endpoint RAG"""
    query_data = {
        "query": "What is Python?",
        "namespace": "test-docs"
    }
    response = client.post("/rag/query", json=query_data)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "sources" in data

def test_upload_endpoint():
    """Teste de upload de arquivo"""
    test_file_content = b"This is a test document content."
    files = {"file": ("test.txt", test_file_content, "text/plain")}
    data = {"namespace": "test-upload"}

    response = client.post("/upload", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert "message" in result
    assert "document_id" in result
```

### Testes de Integração

```python
# hayhooks/tests/test_integration.py
import pytest
import redis
from hayhooks.cache_manager import CacheManager

@pytest.fixture
def cache_manager():
    """Fixture para testes de cache"""
    return CacheManager()

def test_cache_embedding_flow(cache_manager):
    """Teste do fluxo completo de cache de embeddings"""
    test_text = "Test text for embedding"
    test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Verificar que não existe no cache
    cached = cache_manager.get_embedding(test_text)
    assert cached is None

    # Armazenar no cache
    cache_manager.store_embedding(test_text, test_embedding)

    # Verificar que foi armazenado
    cached = cache_manager.get_embedding(test_text)
    assert cached == test_embedding

@pytest.mark.integration
def test_full_rag_pipeline():
    """Teste do pipeline RAG completo"""
    # Esse teste requer serviços externos rodando
    # Use apenas quando necessário
    pass
```

### Testes com Mock

```python
# hayhooks/tests/test_mocked.py
from unittest.mock import patch, MagicMock
import pytest

@patch('hayhooks.cached_embedder.OpenAI')
def test_embedder_with_mock(mock_openai):
    """Teste do embedder com OpenAI mockado"""
    # Configurar mock
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_client.embeddings.create.return_value = mock_response
    mock_openai.return_value = mock_client

    # Importar após o patch
    from hayhooks.cached_embedder import CachedEmbedder

    embedder = CachedEmbedder()
    result = embedder.run("test text")

    assert "embedding" in result
    assert len(result["embedding"]) == 3
```

---

## 🐛 Debugging

### Configuração de Debug

#### Environment Variables para Debug

```bash
# .env (desenvolvimento)
ENABLE_DEBUG_LOGS=true
LOG_LEVEL=DEBUG
OPENAI_CHAT_MODEL=gpt-4o-mini  # Modelo mais barato para debug
```

#### Logging Avançado

```python
# hayhooks/utils/logger.py
import logging
import sys
from datetime import datetime

def setup_logger(name: str, level: str = "INFO"):
    """Configurar logger com formatação adequada"""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # Handler para arquivo
    file_handler = logging.FileHandler('debug.log')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Usar em modules
logger = setup_logger(__name__, os.getenv("LOG_LEVEL", "INFO"))
```

### Debugging Específico

#### Debug Pipeline

```python
# hayhooks/pipelines/debug_rag.py
import logging
from haystack import Pipeline, component

logger = logging.getLogger(__name__)

@component
class DebugComponent:
    """Componente para debug de pipeline"""

    @component.output_types(debug_info=dict, passthrough=dict)
    def run(self, **kwargs) -> dict:
        logger.info(f"Debug Component Input: {kwargs}")

        debug_info = {
            "component": "DebugComponent",
            "timestamp": datetime.now().isoformat(),
            "input_keys": list(kwargs.keys()),
            "input_types": {k: type(v).__name__ for k, v in kwargs.items()}
        }

        logger.info(f"Debug Info: {debug_info}")

        return {
            "debug_info": debug_info,
            "passthrough": kwargs
        }

def create_debug_pipeline():
    """Pipeline com componentes de debug"""
    pipeline = Pipeline()

    pipeline.add_component("debug_start", DebugComponent())
    pipeline.add_component("embedder", cached_embedder)
    pipeline.add_component("debug_embed", DebugComponent())
    pipeline.add_component("retriever", pinecone_retriever)
    pipeline.add_component("debug_retrieve", DebugComponent())
    pipeline.add_component("generator", openai_generator)
    pipeline.add_component("debug_end", DebugComponent())

    # Conectar com debug entre cada etapa
    pipeline.connect("debug_start.passthrough", "embedder.text")
    pipeline.connect("embedder", "debug_embed.embedding")
    pipeline.connect("debug_embed.passthrough", "retriever")
    pipeline.connect("retriever", "debug_retrieve.documents")
    pipeline.connect("debug_retrieve.passthrough", "generator")
    pipeline.connect("generator", "debug_end.response")

    return pipeline
```

#### Debug Cache

```python
# hayhooks/cache_manager.py (versão debug)
class DebugCacheManager(CacheManager):
    """Cache manager com debug extra"""

    def get_embedding(self, text_hash: str):
        logger.debug(f"Cache GET: {text_hash}")
        result = super().get_embedding(text_hash)

        if result:
            logger.debug(f"Cache HIT: {text_hash}")
        else:
            logger.debug(f"Cache MISS: {text_hash}")

        return result

    def store_embedding(self, text_hash: str, embedding: List[float]):
        logger.debug(f"Cache STORE: {text_hash} (size: {len(embedding)})")
        return super().store_embedding(text_hash, embedding)
```

### Performance Profiling

```python
# hayhooks/utils/profiler.py
import time
import functools
import cProfile
import pstats
from typing import Callable

def profile_function(func: Callable):
    """Decorator para profile de função"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        pr.disable()

        # Salvar stats
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.dump_stats(f'profile_{func.__name__}_{int(time.time())}.prof')

        logger.info(f"{func.__name__} took {end_time - start_time:.4f} seconds")

        return result
    return wrapper

# Usar no código
@profile_function
def expensive_operation():
    # Operação custosa
    pass
```

---

## 📏 Best Practices

### 1. Code Style

#### Python Style Guide

```python
# Seguir PEP 8
# Usar type hints
from typing import List, Dict, Optional, Union

def process_documents(
    documents: List[Document],
    namespace: str,
    chunk_size: int = 1000
) -> Dict[str, Union[str, int]]:
    """
    Processa documentos e retorna estatísticas.

    Args:
        documents: Lista de documentos para processar
        namespace: Namespace do Pinecone
        chunk_size: Tamanho dos chunks

    Returns:
        Dict com estatísticas do processamento
    """
    # Implementação
    pass

# Usar dataclasses para estruturas de dados
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    document_count: int
    chunk_count: int
    processing_time: float
    errors: List[str]
```

#### Error Handling

```python
# Criar exceptions customizadas
class RAGSystemError(Exception):
    """Base exception para o sistema RAG"""
    pass

class EmbeddingError(RAGSystemError):
    """Erro relacionado a embeddings"""
    pass

class VectorStoreError(RAGSystemError):
    """Erro relacionado ao vector store"""
    pass

# Usar try/except apropriadamente
async def safe_rag_query(query: str, namespace: str):
    try:
        result = await rag_pipeline.run(query=query, namespace=namespace)
        return result
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise EmbeddingError(f"Failed to generate embedding: {e}")
    except PineconeError as e:
        logger.error(f"Pinecone error: {e}")
        raise VectorStoreError(f"Vector search failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in RAG query: {e}")
        raise RAGSystemError(f"RAG query failed: {e}")
```

### 2. Configuration Management

```python
# hayhooks/config.py
from pydantic import BaseSettings, validator
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str
    pinecone_api_key: str
    anthropic_api_key: Optional[str] = None

    # Database
    redis_password: str
    redis_host: str = "redis"
    redis_port: int = 6379

    # Models
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # Features
    enable_debug_logs: bool = False
    enable_cache: bool = True
    cache_ttl: int = 3600

    @validator('openai_api_key')
    def validate_openai_key(cls, v):
        if not v.startswith('sk-'):
            raise ValueError('Invalid OpenAI API key format')
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instância global
settings = Settings()
```

### 3. Async Programming

```python
# Usar async/await apropriadamente
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def process_multiple_documents(documents: List[Document]):
    """Processar múltiplos documentos de forma assíncrona"""

    # Para I/O bound operations
    async def process_single_doc(doc: Document):
        embedding = await get_embedding_async(doc.content)
        await store_in_vector_db_async(doc, embedding)
        return doc.id

    # Executar em paralelo
    tasks = [process_single_doc(doc) for doc in documents]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Processar resultados
    successful = [r for r in results if not isinstance(r, Exception)]
    errors = [r for r in results if isinstance(r, Exception)]

    return {
        "successful": len(successful),
        "errors": len(errors),
        "processed_ids": successful
    }

# Para CPU bound operations, use ThreadPoolExecutor
async def cpu_intensive_task(data):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, heavy_computation, data)
    return result
```

---

## ⚡ Performance Guidelines

### 1. Caching Strategy

```python
# Implementar cache em múltiplas camadas
class OptimizedCacheManager:
    def __init__(self):
        self.memory_cache = {}  # L1 - Memory
        self.redis_cache = redis_client  # L2 - Redis
        self.memory_max_size = 1000

    async def get_with_fallback(self, key: str):
        # L1 Cache
        if key in self.memory_cache:
            return self.memory_cache[key]

        # L2 Cache
        redis_value = await self.redis_cache.get(key)
        if redis_value:
            # Populate L1
            if len(self.memory_cache) < self.memory_max_size:
                self.memory_cache[key] = redis_value
            return redis_value

        return None

    async def set_multi_level(self, key: str, value: any, ttl: int = 3600):
        # Set in both levels
        if len(self.memory_cache) < self.memory_max_size:
            self.memory_cache[key] = value

        await self.redis_cache.setex(key, ttl, value)
```

### 2. Database Optimization

```python
# Batch operations para Pinecone
async def batch_upsert_vectors(vectors: List[dict], batch_size: int = 100):
    """Upsert vectors em batches para melhor performance"""

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]

        try:
            index.upsert(vectors=batch)
            logger.info(f"Upserted batch {i//batch_size + 1}")

        except Exception as e:
            logger.error(f"Failed to upsert batch {i//batch_size + 1}: {e}")
            # Retry individual vectors in batch
            for vector in batch:
                try:
                    index.upsert(vectors=[vector])
                except Exception as ve:
                    logger.error(f"Failed to upsert vector {vector['id']}: {ve}")
```

### 3. Memory Management

```python
# Gerenciamento de memória para grandes datasets
import gc
from typing import Generator

def chunk_documents(documents: List[Document], chunk_size: int = 100) -> Generator:
    """Process documents in chunks para evitar memory issues"""

    for i in range(0, len(documents), chunk_size):
        chunk = documents[i:i + chunk_size]
        yield chunk

        # Force garbage collection após cada chunk
        gc.collect()

async def process_large_dataset(documents: List[Document]):
    """Processar dataset grande sem sobrecarregar memória"""

    processed_count = 0

    for chunk in chunk_documents(documents):
        await process_document_chunk(chunk)
        processed_count += len(chunk)

        logger.info(f"Processed {processed_count}/{len(documents)} documents")

        # Opcional: adicionar delay para não sobrecarregar APIs
        await asyncio.sleep(0.1)
```

### 4. Monitoring & Metrics

```python
# hayhooks/utils/metrics.py
import time
from functools import wraps
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)

    def time_function(self, func_name: str = None):
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    self.counters[f"{func_name or func.__name__}_success"] += 1
                    return result
                except Exception as e:
                    self.counters[f"{func_name or func.__name__}_error"] += 1
                    raise
                finally:
                    duration = time.time() - start_time
                    self.metrics[func_name or func.__name__].append(duration)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.counters[f"{func_name or func.__name__}_success"] += 1
                    return result
                except Exception as e:
                    self.counters[f"{func_name or func.__name__}_error"] += 1
                    raise
                finally:
                    duration = time.time() - start_time
                    self.metrics[func_name or func.__name__].append(duration)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    def get_stats(self):
        stats = {}
        for func_name, times in self.metrics.items():
            if times:
                stats[func_name] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }

        stats["counters"] = dict(self.counters)
        return stats

# Instância global
monitor = PerformanceMonitor()

# Usar nos endpoints críticos
@monitor.time_function("rag_query")
async def rag_query_endpoint(query: str):
    # Implementação
    pass
```

---

**Status**: ✅ Guia completo de desenvolvimento atualizado
