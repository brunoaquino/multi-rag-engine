# ⚙️ Configuration Guide

Guia completo de configuração do Haystack RAG System.

## 📋 Índice

- [Configuração Básica](#configuração-básica)
- [Variáveis de Ambiente](#variáveis-de-ambiente)
- [Configuração dos Serviços](#configuração-dos-serviços)
- [Configuração de Produção](#configuração-de-produção)
- [Configuração Avançada](#configuração-avançada)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Configuração Básica

### 1. Clonar o Repositório

```bash
git clone <repository-url>
cd haystack-rag-system
```

### 2. Configurar Variáveis de Ambiente

```bash
# Copiar arquivo de exemplo
cp local.env.example .env

# Editar com suas credenciais
nano .env
```

### 3. Configurar Credenciais

Edite o arquivo `.env` com suas credenciais:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key-here

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX_NAME=haystack-docs
PINECONE_NAMESPACE=documents

# Redis Configuration
REDIS_PASSWORD=your-redis-password-here
REDIS_DB=0

# Application Configuration
HAYHOOKS_PORT=8000
WEBUI_PORT=3000
REDIS_PORT=6379

# Optional: Anthropic for alternative LLM
ANTHROPIC_API_KEY=your-anthropic-key-here

# Security (opcional para produção)
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here
```

### 4. Iniciar Sistema

```bash
# Build e start todos os serviços
docker-compose up -d

# Verificar status
docker-compose ps

# Ver logs
docker-compose logs -f
```

---

## 🔐 Variáveis de Ambiente

### Core APIs

#### OpenAI (Obrigatório)

```bash
# Chave da API OpenAI
OPENAI_API_KEY=sk-proj-...
# Modelo padrão para embeddings
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# Modelo padrão para geração
OPENAI_CHAT_MODEL=gpt-4o-mini
# Organização (opcional)
OPENAI_ORG_ID=org-...
```

#### Pinecone (Obrigatório)

```bash
# Chave da API Pinecone
PINECONE_API_KEY=pcsk_...
# Nome do índice
PINECONE_INDEX_NAME=haystack-docs
# Namespace padrão
PINECONE_NAMESPACE=documents
# Environment (se aplicável)
PINECONE_ENVIRONMENT=us-east-1-aws
```

#### Anthropic (Opcional)

```bash
# Chave da API Anthropic
ANTHROPIC_API_KEY=sk-ant-...
# Modelo padrão
ANTHROPIC_MODEL=claude-3-haiku-20240307
```

### Cache & Database

#### Redis

```bash
# Senha do Redis
REDIS_PASSWORD=strongPassword123
# Banco de dados
REDIS_DB=0
# Host (geralmente 'redis' no Docker)
REDIS_HOST=redis
# Porta
REDIS_PORT=6379
# TTL padrão para cache (segundos)
REDIS_TTL=3600
```

### Application Settings

#### Ports & Networking

```bash
# Porta do Hayhooks API
HAYHOOKS_PORT=8000
# Porta do OpenWebUI
WEBUI_PORT=3000
# Porta do Redis
REDIS_PORT=6379
# Porta do Admin Dashboard (se habilitado)
ADMIN_PORT=8502
```

#### Security

```bash
# Chave secreta da aplicação
SECRET_KEY=supersecretkey12345
# Chave para JWT tokens
JWT_SECRET_KEY=jwtsecretkey12345
# Algoritmo JWT
JWT_ALGORITHM=HS256
# Expiração do token (horas)
JWT_EXPIRATION_HOURS=24
```

#### Features

```bash
# Habilitar cache de embeddings
ENABLE_EMBEDDING_CACHE=true
# Habilitar cache de queries
ENABLE_QUERY_CACHE=true
# Habilitar logs detalhados
ENABLE_DEBUG_LOGS=false
# Número máximo de documentos por upload
MAX_DOCUMENTS_PER_UPLOAD=10
# Tamanho máximo de arquivo (MB)
MAX_FILE_SIZE_MB=50
```

---

## 🛠️ Configuração dos Serviços

### Hayhooks API

#### Configuração Interna (`hayhooks/direct_api.py`)

```python
# Configurações carregadas do ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "haystack-docs")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Configuração do Redis
redis_client = Redis(
    host='redis',
    port=6379,
    password=REDIS_PASSWORD,
    decode_responses=True
)

# Configuração do Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
)
```

#### Docker Configuration

```yaml
# docker-compose.yml
hayhooks:
  build:
    context: ./hayhooks
    dockerfile: Dockerfile
  environment:
    - OPENAI_API_KEY=${OPENAI_API_KEY}
    - PINECONE_API_KEY=${PINECONE_API_KEY}
    - PINECONE_INDEX_NAME=${PINECONE_INDEX_NAME}
    - PINECONE_NAMESPACE=${PINECONE_NAMESPACE}
    - REDIS_PASSWORD=${REDIS_PASSWORD}
  ports:
    - "${HAYHOOKS_PORT:-8000}:8000"
  volumes:
    - ./data:/app/data
  depends_on:
    - redis
```

### OpenWebUI

#### Environment Variables

```yaml
open-webui:
  image: ghcr.io/open-webui/open-webui:main
  environment:
    # Backend URL
    - OPENAI_API_BASE_URL=http://hayhooks:8000/v1
    - OPENAI_API_KEY=${OPENAI_API_KEY}

    # Interface customization
    - WEBUI_NAME=Haystack RAG
    - WEBUI_URL=http://localhost:3000
    - WEBUI_SECRET_KEY=${SECRET_KEY}

    # Features
    - ENABLE_SIGNUP=false
    - ENABLE_LOGIN_FORM=true
    - DEFAULT_USER_ROLE=user
  ports:
    - "${WEBUI_PORT:-3000}:8080"
```

### Redis

#### Basic Configuration

```yaml
redis:
  image: redis:7.2-alpine
  command: >
    redis-server 
    --requirepass ${REDIS_PASSWORD}
    --appendonly yes
    --appendfsync everysec
  environment:
    - REDIS_PASSWORD=${REDIS_PASSWORD}
  ports:
    - "${REDIS_PORT:-6379}:6379"
  volumes:
    - redis_data:/data
```

#### Advanced Redis Configuration (`redis.conf`)

```bash
# redis.conf
# Memory
maxmemory 1gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Security
requirepass ${REDIS_PASSWORD}
protected-mode yes

# Performance
tcp-keepalive 300
timeout 0
```

---

## 🏭 Configuração de Produção

### 1. Environment-Specific Configs

#### Desenvolvimento (`.env.development`)

```bash
# Debug enabled
ENABLE_DEBUG_LOGS=true
REDIS_TTL=300  # 5 minutos
MAX_FILE_SIZE_MB=10

# Local services
REDIS_HOST=localhost
PINECONE_NAMESPACE=dev-docs
```

#### Produção (`.env.production`)

```bash
# Debug disabled
ENABLE_DEBUG_LOGS=false
REDIS_TTL=3600  # 1 hora
MAX_FILE_SIZE_MB=100

# Production services
REDIS_HOST=redis-cluster.internal
PINECONE_NAMESPACE=prod-docs

# Security
JWT_EXPIRATION_HOURS=2
ENABLE_RATE_LIMITING=true
```

### 2. Docker Compose Override

#### `docker-compose.prod.yml`

```yaml
version: "3.8"

services:
  hayhooks:
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2g
          cpus: "1.0"
        reservations:
          memory: 1g
          cpus: "0.5"

  redis:
    deploy:
      resources:
        limits:
          memory: 1g
          cpus: "0.5"

  open-webui:
    environment:
      - WEBUI_SECRET_KEY=${PRODUCTION_SECRET_KEY}
    deploy:
      resources:
        limits:
          memory: 1g
          cpus: "0.5"
```

### 3. Nginx Reverse Proxy

#### `nginx.conf`

```nginx
upstream hayhooks_backend {
    least_conn;
    server hayhooks:8000;
}

upstream webui_backend {
    server open-webui:8080;
}

server {
    listen 80;
    server_name your-domain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL configuration
    ssl_certificate /etc/ssl/certs/your-cert.pem;
    ssl_certificate_key /etc/ssl/private/your-key.pem;

    # API routes
    location /api/ {
        proxy_pass http://hayhooks_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # WebUI routes
    location / {
        proxy_pass http://webui_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
}
```

---

## 🔧 Configuração Avançada

### 1. Custom Embeddings Model

#### OpenAI Alternative Model

```python
# hayhooks/cached_embedder.py
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

class CachedEmbedder:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = EMBEDDING_MODEL
        self.dimensions = 3072 if "large" in self.model else 1536
```

#### Sentence Transformers (Local)

```python
# Para usar modelos locais
from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
```

### 2. Multiple LLM Providers

#### Configuration

```python
# hayhooks/model_config.py
LLM_PROVIDERS = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    },
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "models": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"]
    }
}

def get_llm_client(provider: str, model: str):
    if provider == "openai":
        return OpenAI(api_key=LLM_PROVIDERS["openai"]["api_key"])
    elif provider == "anthropic":
        return Anthropic(api_key=LLM_PROVIDERS["anthropic"]["api_key"])
```

### 3. Custom Vector Store

#### Weaviate Alternative

```python
# hayhooks/vector_stores/weaviate_store.py
import weaviate

class WeaviateStore:
    def __init__(self):
        self.client = weaviate.Client(
            url=os.getenv("WEAVIATE_URL"),
            auth_client_secret=weaviate.AuthApiKey(
                api_key=os.getenv("WEAVIATE_API_KEY")
            )
        )

    def add_documents(self, documents: List[Document]):
        # Implementation for Weaviate
        pass
```

### 4. Advanced Caching

#### Multi-Level Cache

```python
# hayhooks/cache/multi_level.py
from functools import lru_cache
import redis
import pickle

class MultiLevelCache:
    def __init__(self):
        self.redis = redis.Redis(
            host='redis',
            password=os.getenv('REDIS_PASSWORD')
        )
        self.l1_cache = {}
        self.l1_max_size = 1000

    @lru_cache(maxsize=100)
    def get_embedding(self, text_hash: str):
        # L1 Cache (memory)
        if text_hash in self.l1_cache:
            return self.l1_cache[text_hash]

        # L2 Cache (Redis)
        cached = self.redis.get(f"embedding:{text_hash}")
        if cached:
            embedding = pickle.loads(cached)
            self.l1_cache[text_hash] = embedding
            return embedding

        return None
```

### 5. Custom Pipeline Components

#### Custom Retriever

```python
# hayhooks/components/custom_retriever.py
from haystack import component, Document
from typing import List, Dict, Any

@component
class CustomRetriever:
    def __init__(self, vector_store, similarity_threshold: float = 0.8):
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ) -> Dict[str, List[Document]]:
        # Custom retrieval logic
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Filter by similarity threshold
        filtered_results = [
            r for r in results
            if r.score >= self.similarity_threshold
        ]

        documents = [
            Document(
                content=r.metadata.get('text', ''),
                meta=r.metadata
            )
            for r in filtered_results
        ]

        return {"documents": documents}
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. API Keys não funcionando

```bash
# Verificar se as variáveis estão sendo carregadas
docker-compose exec hayhooks env | grep API_KEY

# Testar API keys individualmente
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

#### 2. Redis Connection Failed

```bash
# Verificar logs do Redis
docker-compose logs redis

# Testar conexão
docker-compose exec redis redis-cli -a $REDIS_PASSWORD ping

# Verificar senha
docker-compose exec hayhooks python -c "
import redis
r = redis.Redis(host='redis', password='$REDIS_PASSWORD')
print(r.ping())
"
```

#### 3. Pinecone Index não encontrado

```bash
# Verificar índices existentes
python -c "
import pinecone
pinecone.init(api_key='$PINECONE_API_KEY')
print(pinecone.list_indexes())
"

# Criar índice se necessário
python -c "
import pinecone
pinecone.init(api_key='$PINECONE_API_KEY')
pinecone.create_index(
    name='haystack-docs',
    dimension=1536,
    metric='cosine'
)
"
```

#### 4. Docker Memory Issues

```bash
# Verificar uso de memória
docker stats

# Aumentar limites no docker-compose.yml
services:
  hayhooks:
    deploy:
      resources:
        limits:
          memory: 4g
```

#### 5. File Upload Issues

```bash
# Verificar permissões do diretório
ls -la ./data/

# Corrigir permissões
sudo chown -R 1000:1000 ./data/
chmod -R 755 ./data/
```

### Debug Mode

#### Habilitar Debug Logs

```bash
# No .env
ENABLE_DEBUG_LOGS=true
LOG_LEVEL=DEBUG

# Restart services
docker-compose restart hayhooks
```

#### Verificar Logs

```bash
# Logs de todos os serviços
docker-compose logs -f

# Logs específicos
docker-compose logs -f hayhooks
docker-compose logs -f redis
docker-compose logs -f open-webui
```

### Health Checks

#### API Status

```bash
curl http://localhost:8000/health
```

#### Cache Status

```bash
curl http://localhost:8000/cache/stats
```

#### RAG Pipeline Status

```bash
curl http://localhost:8000/rag/info
```

---

**Status**: ✅ Guia completo de configuração do sistema
