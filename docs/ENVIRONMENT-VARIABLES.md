# 🔧 Environment Variables Reference Guide

Este guia documenta todas as variáveis de ambiente configuráveis no sistema Haystack RAG, incluindo valores padrão, intervalos aceitáveis e impacto no comportamento do sistema.

> 📖 **Para setup básico**: [README.md](../README.md)  
> 🚀 **Para deploy detalhado**: [DEPLOYMENT.md](DEPLOYMENT.md)  
> 🔧 **Para troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [API Keys (Obrigatórias)](#-api-keys-obrigatórias)
- [Configurações de Banco de Dados](#-configurações-de-banco-de-dados)
- [Portas e Networking](#-portas-e-networking)
- [Configurações de AI/ML](#-configurações-de-aiml)
- [Cache e Performance](#-cache-e-performance)
- [Segurança e Autenticação](#-segurança-e-autenticação)
- [Logging e Debug](#-logging-e-debug)
- [Configurações de Upload](#-configurações-de-upload)
- [Configurações de Interface](#-configurações-de-interface)
- [Templates de .env](#-templates-de-env)

## 🎯 Visão Geral

### Prioridade das Configurações

| Prioridade         | Categoria       | Descrição                      |
| ------------------ | --------------- | ------------------------------ |
| **🔴 Críticas**    | API Keys        | Sistema não funciona sem essas |
| **🟡 Importantes** | Database/Cache  | Afetam funcionalidade core     |
| **🟢 Opcionais**   | Performance     | Otimizações e personalizações  |
| **🔵 Debug**       | Desenvolvimento | Logs e debugging               |

### Locais de Configuração

```bash
# Desenvolvimento local
.env                    # Variáveis principais

# Docker Compose
docker-compose.yml      # Overrides específicos

# Produção
/etc/environment        # System-wide
~/.bashrc              # User-specific
```

## 🔑 API Keys (Obrigatórias)

### OpenAI API

```bash
# Configuração
OPENAI_API_KEY=sk-proj-your-key-here

# Validação
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

| Propriedade        | Valor                                       |
| ------------------ | ------------------------------------------- |
| **Formato**        | `sk-proj-[A-Za-z0-9]+` (51 chars)           |
| **Onde obter**     | https://platform.openai.com/api-keys        |
| **Modelos**        | GPT-4o, GPT-4o-mini, text-embedding-3-small |
| **Custo estimado** | $10-50/mês para uso médio                   |
| **Rate limits**    | 3,500 RPM (Tier 1)                          |

### Pinecone Vector Database

```bash
# Configuração
PINECONE_API_KEY=pcsk_your-pinecone-key
PINECONE_ENVIRONMENT=gcp-starter
PINECONE_INDEX_NAME=haystack-docs
```

| Variável               | Padrão          | Descrição           | Valores Aceitos                      |
| ---------------------- | --------------- | ------------------- | ------------------------------------ |
| `PINECONE_API_KEY`     | _obrigatório_   | API key do Pinecone | UUID format                          |
| `PINECONE_ENVIRONMENT` | `gcp-starter`   | Região/ambiente     | `gcp-starter`, `us-east-1-aws`, etc. |
| `PINECONE_INDEX_NAME`  | `haystack-docs` | Nome do índice      | [a-z0-9-]+                           |

#### Configuração do Índice

```python
# Criação automática se não existir
pinecone.create_index(
    name=PINECONE_INDEX_NAME,
    dimension=1536,  # OpenAI embeddings
    metric="cosine",
    pod_type="s1.x1"
)
```

### Anthropic Claude (Opcional)

```bash
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

| Propriedade | Valor                               |
| ----------- | ----------------------------------- |
| **Formato** | `sk-ant-[A-Za-z0-9]+`               |
| **Modelos** | Claude 3.5 Sonnet, Claude 3.5 Haiku |
| **Uso**     | Alternativa ao OpenAI               |

## 🗄️ Configurações de Banco de Dados

### Redis Cache

```bash
# Configuração básica
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redis123
REDIS_DB=0

# Configurações avançadas
REDIS_MAX_CONNECTIONS=20
REDIS_SOCKET_TIMEOUT=5
REDIS_CONNECTION_TIMEOUT=10
```

| Variável                | Padrão     | Descrição          | Impacto     |
| ----------------------- | ---------- | ------------------ | ----------- |
| `REDIS_HOST`            | `redis`    | Hostname do Redis  | Conexão     |
| `REDIS_PORT`            | `6379`     | Porta do Redis     | 1-65535     |
| `REDIS_PASSWORD`        | `redis123` | Senha Redis        | Segurança   |
| `REDIS_DB`              | `0`        | Database Redis     | 0-15        |
| `REDIS_MAX_CONNECTIONS` | `20`       | Pool de conexões   | Performance |
| `REDIS_SOCKET_TIMEOUT`  | `5`        | Timeout socket (s) | Latência    |

#### Configurações de Cache

```bash
# TTL por tipo de cache
CACHE_TTL_EMBEDDINGS=86400      # 24 horas
CACHE_TTL_QUERIES=3600          # 1 hora
CACHE_TTL_DOCUMENTS=43200       # 12 horas

# Configurações de memória
REDIS_MAXMEMORY=256mb
REDIS_MAXMEMORY_POLICY=allkeys-lru
```

## 🌐 Portas e Networking

### Portas Principais

```bash
# Serviços principais
HAYHOOKS_PORT=8000          # API Backend
HAYHOOKS_PIPELINE_PORT=1416 # Pipelines Haystack
WEBUI_PORT=3000             # OpenWebUI Interface
STREAMLIT_PORT=8501         # Upload Frontend
REDIS_PORT=6379             # Redis Cache

# Proxy e Load Balancing
NGINX_HTTP_PORT=80
NGINX_HTTPS_PORT=443
```

| Serviço                | Variável                 | Padrão | Faixa     | Descrição         |
| ---------------------- | ------------------------ | ------ | --------- | ----------------- |
| **Hayhooks API**       | `HAYHOOKS_PORT`          | `8000` | 8000-8999 | API principal     |
| **Hayhooks Pipelines** | `HAYHOOKS_PIPELINE_PORT` | `1416` | 1400-1499 | Pipelines nativos |
| **OpenWebUI**          | `WEBUI_PORT`             | `3000` | 3000-3099 | Interface web     |
| **Streamlit**          | `STREAMLIT_PORT`         | `8501` | 8500-8599 | Upload UI         |
| **Redis**              | `REDIS_PORT`             | `6379` | 6379      | Cache (padrão)    |

### Configurações de Rede

```bash
# Host binding
HAYHOOKS_HOST=0.0.0.0
WEBUI_HOST=0.0.0.0

# External URLs (produção)
PUBLIC_API_URL=https://api.yourdomain.com
PUBLIC_WEBUI_URL=https://chat.yourdomain.com
PUBLIC_UPLOAD_URL=https://upload.yourdomain.com
```

## 🤖 Configurações de AI/ML

### Modelos e Providers

```bash
# Modelo padrão para embeddings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Modelo padrão para chat/geração
CHAT_MODEL=gpt-4o-mini
FALLBACK_MODEL=gpt-3.5-turbo

# Configurações de geração
DEFAULT_MAX_TOKENS=1000
DEFAULT_TEMPERATURE=0.1
DEFAULT_TOP_P=1.0
```

| Variável               | Padrão                   | Valores           | Impacto            |
| ---------------------- | ------------------------ | ----------------- | ------------------ |
| `EMBEDDING_MODEL`      | `text-embedding-3-small` | OpenAI embeddings | Qualidade da busca |
| `EMBEDDING_DIMENSIONS` | `1536`                   | 1536, 3072        | Precision vs Speed |
| `CHAT_MODEL`           | `gpt-4o-mini`            | GPT models        | Qualidade resposta |
| `DEFAULT_MAX_TOKENS`   | `1000`                   | 100-4000          | Tamanho resposta   |
| `DEFAULT_TEMPERATURE`  | `0.1`                    | 0.0-2.0           | Criatividade       |

### Configurações RAG

```bash
# Processamento de documentos
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CHUNKS_PER_DOC=50

# Retrieval
DEFAULT_TOP_K=5
SIMILARITY_THRESHOLD=0.7
RERANK_TOP_K=3

# Namespace padrão
DEFAULT_NAMESPACE=documents
```

| Variável               | Padrão | Faixa    | Descrição                 |
| ---------------------- | ------ | -------- | ------------------------- |
| `CHUNK_SIZE`           | `1000` | 200-2000 | Tamanho chunks de texto   |
| `CHUNK_OVERLAP`        | `200`  | 50-500   | Sobreposição entre chunks |
| `DEFAULT_TOP_K`        | `5`    | 1-20     | Documentos recuperados    |
| `SIMILARITY_THRESHOLD` | `0.7`  | 0.0-1.0  | Threshold de similaridade |

## ⚡ Cache e Performance

### Configurações de Cache

```bash
# Cache application-level
ENABLE_CACHE=true
CACHE_DEFAULT_TTL=3600

# Cache por tipo
CACHE_EMBEDDINGS=true
CACHE_QUERIES=true
CACHE_DOCUMENTS=true

# Performance
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30
CONNECTION_POOL_SIZE=20
```

| Variável                  | Padrão | Descrição              | Impacto           |
| ------------------------- | ------ | ---------------------- | ----------------- |
| `ENABLE_CACHE`            | `true` | Habilitar cache global | Performance geral |
| `CACHE_DEFAULT_TTL`       | `3600` | TTL padrão (segundos)  | Refresh rate      |
| `MAX_CONCURRENT_REQUESTS` | `10`   | Requests simultâneos   | Throughput        |
| `REQUEST_TIMEOUT`         | `30`   | Timeout requests (s)   | UX                |

### Memory Management

```bash
# Limites de memória
MAX_MEMORY_USAGE=2048M
EMBEDDINGS_CACHE_SIZE=500M
QUERY_CACHE_SIZE=100M

# Garbage collection
GC_THRESHOLD=0.8
CLEANUP_INTERVAL=300
```

## 🔒 Segurança e Autenticação

### OpenWebUI Security

```bash
# Autenticação
WEBUI_SECRET_KEY=your-secret-key-here
ENABLE_SIGNUP=false
ENABLE_LOGIN_FORM=true
DEFAULT_USER_ROLE=user

# JWT
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30
```

| Variável                          | Padrão        | Descrição         | Segurança           |
| --------------------------------- | ------------- | ----------------- | ------------------- |
| `WEBUI_SECRET_KEY`                | _obrigatório_ | Chave para JWT    | Alta                |
| `ENABLE_SIGNUP`                   | `false`       | Permitir registro | Controle acesso     |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `60`          | Expiração token   | Balance UX/security |

### CORS e Headers

```bash
# CORS
CORS_ALLOW_ORIGINS="http://localhost:3000,http://localhost:8501"
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS="GET,POST,PUT,DELETE"

# Security headers
SECURITY_HEADERS=true
HSTS_MAX_AGE=31536000
```

### Rate Limiting

```bash
# Rate limits
RATE_LIMIT_UPLOAD=10/minute
RATE_LIMIT_QUERY=60/minute
RATE_LIMIT_CHAT=30/minute

# IP-based limits
MAX_REQUESTS_PER_IP=1000/day
BLOCK_DURATION=3600
```

## 📊 Logging e Debug

### Níveis de Log

```bash
# Configuração principal
LOG_LEVEL=INFO
LOG_FORMAT=json

# Logs por componente
HAYHOOKS_LOG_LEVEL=INFO
OPENWEBUI_LOG_LEVEL=WARNING
REDIS_LOG_LEVEL=NOTICE

# Arquivos de log
LOG_FILE_PATH=/var/log/haystack
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=5
LOG_ROTATION=daily
```

| Level      | Descrição                       | Uso             |
| ---------- | ------------------------------- | --------------- |
| `DEBUG`    | Logs detalhados                 | Desenvolvimento |
| `INFO`     | Informações gerais              | Produção normal |
| `WARNING`  | Avisos não críticos             | Monitoramento   |
| `ERROR`    | Erros que afetam funcionalidade | Alertas         |
| `CRITICAL` | Erros críticos do sistema       | Emergências     |

### Debug Features

```bash
# Debug modes
DEBUG_MODE=false
VERBOSE_LOGGING=false
TRACE_REQUESTS=false

# Performance profiling
ENABLE_PROFILING=false
PROFILE_SLOW_QUERIES=true
SLOW_QUERY_THRESHOLD=2.0

# Development
RELOAD_ON_CHANGE=false
ENABLE_API_DOCS=true
```

## 📤 Configurações de Upload

### Limites de Arquivo

```bash
# Tamanhos máximos
MAX_UPLOAD_SIZE=100MB
MAX_FILE_SIZE_PDF=50MB
MAX_FILE_SIZE_TXT=10MB
MAX_FILE_SIZE_DOCX=25MB

# Limites por usuário
MAX_FILES_PER_USER=100
MAX_STORAGE_PER_USER=1GB
UPLOAD_RATE_LIMIT=10/hour
```

| Variável             | Padrão    | Máximo     | Descrição            |
| -------------------- | --------- | ---------- | -------------------- |
| `MAX_UPLOAD_SIZE`    | `100MB`   | `1GB`      | Tamanho por upload   |
| `MAX_FILES_PER_USER` | `100`     | `1000`     | Arquivos por usuário |
| `UPLOAD_RATE_LIMIT`  | `10/hour` | `100/hour` | Taxa de upload       |

### Processamento

```bash
# Processamento de documentos
PROCESSING_TIMEOUT=300
PARALLEL_PROCESSING=true
MAX_PROCESSING_WORKERS=4

# Formatos suportados
SUPPORTED_FORMATS="pdf,txt,docx,md,csv,json"
AUTO_DETECT_ENCODING=true
FALLBACK_ENCODING=utf-8
```

## 🖥️ Configurações de Interface

### OpenWebUI Customization

```bash
# Branding
WEBUI_NAME="Haystack RAG"
WEBUI_FAVICON="/static/favicon.ico"
CUSTOM_CSS_URL=""

# Features
ENABLE_RAG=true
ENABLE_WEB_SEARCH=false
ENABLE_IMAGE_GENERATION=false

# Default settings
DEFAULT_MODELS="haystack-rag"
DEFAULT_SYSTEM_PROMPT=""
```

### Streamlit Configuration

```bash
# Streamlit específico
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_THEME_BASE=light

# Upload UI
STREAMLIT_MAX_UPLOAD_SIZE=200MB
STREAMLIT_FILE_UPLOADER_ENABLED=true
STREAMLIT_ENABLE_DRAG_DROP=true
```

## 📋 Templates de .env

### Desenvolvimento Local

```bash
# .env.development
# =================

# 🔑 API Keys (obrigatórias)
OPENAI_API_KEY=sk-proj-your-dev-key
PINECONE_API_KEY=pcsk_your-dev-key

# 🗄️ Database
REDIS_PASSWORD=redis123
PINECONE_ENVIRONMENT=gcp-starter
PINECONE_INDEX_NAME=haystack-dev

# 🌐 Portas (padrão)
HAYHOOKS_PORT=8000
WEBUI_PORT=3000
STREAMLIT_PORT=8501
REDIS_PORT=6379

# 🔒 Segurança (dev)
WEBUI_SECRET_KEY=dev-secret-key
ENABLE_SIGNUP=true
DEBUG_MODE=true

# 📊 Logs (verbose)
LOG_LEVEL=DEBUG
VERBOSE_LOGGING=true
ENABLE_API_DOCS=true
```

### Produção

```bash
# .env.production
# ===============

# 🔑 API Keys (produção)
OPENAI_API_KEY=sk-proj-your-prod-key
PINECONE_API_KEY=pcsk_your-prod-key

# 🗄️ Database
REDIS_PASSWORD=very-strong-redis-password
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=haystack-prod

# 🌐 Network
PUBLIC_API_URL=https://api.yourdomain.com
PUBLIC_WEBUI_URL=https://chat.yourdomain.com
CORS_ALLOW_ORIGINS="https://yourdomain.com"

# 🔒 Segurança (restrita)
WEBUI_SECRET_KEY=very-strong-secret-key-256-chars
ENABLE_SIGNUP=false
DEBUG_MODE=false
SECURITY_HEADERS=true

# ⚡ Performance
REDIS_MAX_CONNECTIONS=50
MAX_CONCURRENT_REQUESTS=20
ENABLE_CACHE=true

# 📊 Logs (produção)
LOG_LEVEL=INFO
LOG_FILE_PATH=/var/log/haystack
LOG_ROTATION=daily
```

### Docker Compose

```bash
# .env.docker
# ===========

# 🐳 Docker específico
COMPOSE_PROJECT_NAME=haystack
DOCKER_DEFAULT_PLATFORM=linux/amd64

# 🗄️ Internal networking
REDIS_HOST=redis
PINECONE_HOST=pinecone-proxy

# 🔧 Resource limits
HAYHOOKS_MEMORY_LIMIT=2g
REDIS_MEMORY_LIMIT=512m
WEBUI_MEMORY_LIMIT=1g
```

## 🔧 Validação e Testing

### Script de Validação

```bash
#!/bin/bash
# validate-env.sh

echo "🔍 Validating environment variables..."

# Função para validar variável obrigatória
validate_required() {
    local var_name=$1
    local var_value=${!var_name}

    if [ -z "$var_value" ]; then
        echo "❌ $var_name is required but not set"
        return 1
    else
        echo "✅ $var_name is set"
        return 0
    fi
}

# Função para validar formato
validate_format() {
    local var_name=$1
    local var_value=${!var_name}
    local pattern=$2

    if [[ $var_value =~ $pattern ]]; then
        echo "✅ $var_name format is valid"
        return 0
    else
        echo "❌ $var_name format is invalid"
        return 1
    fi
}

# Validações obrigatórias
validate_required "OPENAI_API_KEY"
validate_required "PINECONE_API_KEY"
validate_required "REDIS_PASSWORD"

# Validações de formato
validate_format "OPENAI_API_KEY" "^sk-proj-[A-Za-z0-9]+$"
validate_format "WEBUI_PORT" "^[0-9]+$"
validate_format "REDIS_PORT" "^[0-9]+$"

# Testes de conectividade
echo "🌐 Testing API connectivity..."

# OpenAI
if curl -s -H "Authorization: Bearer $OPENAI_API_KEY" \
   https://api.openai.com/v1/models >/dev/null; then
    echo "✅ OpenAI API connection successful"
else
    echo "❌ OpenAI API connection failed"
fi

echo "Validation complete!"
```

### Environment Health Check

```bash
#!/bin/bash
# env-health-check.sh

# Check memory settings
if [ "$REDIS_MAXMEMORY" ]; then
    echo "✅ Redis memory limit configured: $REDIS_MAXMEMORY"
else
    echo "⚠️  Redis memory limit not set, using default"
fi

# Check security settings
if [ "$WEBUI_SECRET_KEY" = "dev-secret-key" ]; then
    echo "⚠️  Using development secret key in production"
fi

# Check log levels
if [ "$LOG_LEVEL" = "DEBUG" ] && [ "$DEBUG_MODE" != "true" ]; then
    echo "⚠️  Debug logs enabled but debug mode disabled"
fi
```

## 📚 Troubleshooting Comum

### Problemas Frequentes

| Problema           | Variável Relacionada | Solução                      |
| ------------------ | -------------------- | ---------------------------- |
| API 401            | `OPENAI_API_KEY`     | Verificar formato e validade |
| Connection timeout | `REQUEST_TIMEOUT`    | Aumentar timeout             |
| Cache miss alto    | `CACHE_TTL_*`        | Ajustar TTL values           |
| Upload falha       | `MAX_UPLOAD_SIZE`    | Aumentar limite              |
| Memory error       | `MAX_MEMORY_USAGE`   | Configurar limites           |

### Debug Commands

```bash
# Verificar todas as variáveis
env | grep -E "(OPENAI|PINECONE|REDIS|WEBUI)" | sort

# Testar configuração Redis
redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD ping

# Validar URLs públicas
curl -I $PUBLIC_API_URL/health
curl -I $PUBLIC_WEBUI_URL
```

---

## 🎯 Próximos Passos

1. **Copiar template apropriado** para seu ambiente
2. **Configurar API keys** necessárias
3. **Validar configuração** com scripts fornecidos
4. **Testar conectividade** antes do deploy
5. **Monitorar performance** e ajustar conforme necessário

---

**📅 Última Atualização**: Janeiro 2024  
**🔄 Próxima Revisão**: Trimestral  
**👥 Responsável**: DevOps + Security Team

Para **configuração específica por ambiente**, consulte:

- 🚀 [DEPLOYMENT.md](DEPLOYMENT.md) para deploy detalhado
- 🔧 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) para problemas
- 📖 [README.md](../README.md) para overview geral
