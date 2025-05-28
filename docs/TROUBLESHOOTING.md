# 🔧 Troubleshooting & Monitoring Guide - Haystack RAG System

Este guia fornece soluções detalhadas para problemas comuns, análise de logs, otimização de performance e monitoramento do sistema Haystack RAG.

> 📖 **Para setup inicial**: [README.md](../README.md)  
> 🚀 **Para deploy avançado**: [DEPLOYMENT.md](DEPLOYMENT.md)  
> 🔌 **Para APIs**: [API.md](API.md)

## 📋 Índice

- [Health Checks e Diagnósticos](#-health-checks-e-diagnósticos)
- [Problemas Comuns e Soluções](#-problemas-comuns-e-soluções)
- [Análise de Logs](#-análise-de-logs)
- [Monitoramento e Performance](#-monitoramento-e-performance)
- [Otimização de Sistema](#-otimização-de-sistema)
- [Scripts de Diagnóstico](#-scripts-de-diagnóstico)

## 🩺 Health Checks e Diagnósticos

### Verificação Rápida do Sistema

```bash
#!/bin/bash
# Quick system health check

echo "🔍 Verificando status dos serviços..."

# 1. Docker containers
echo "📦 Docker Containers:"
docker-compose ps

# 2. API Health checks
echo "🔌 API Endpoints:"
curl -s http://localhost:8000/health | jq '.' || echo "❌ API Principal indisponível"
curl -s http://localhost:1416/status | jq '.' || echo "❌ Hayhooks indisponível"
curl -s http://localhost:3000/health || echo "❌ OpenWebUI indisponível"

# 3. Database connections
echo "🗄️ Database Connections:"
docker-compose exec redis redis-cli ping || echo "❌ Redis indisponível"

# 4. External services
echo "🌐 External Services:"
curl -s -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models >/dev/null && echo "✅ OpenAI API" || echo "❌ OpenAI API"

# 5. Disk space
echo "💾 Disk Space:"
df -h | grep -E "/$|/var"
```

### Health Check Endpoints

| Serviço           | URL                   | Status Esperado                                                        |
| ----------------- | --------------------- | ---------------------------------------------------------------------- |
| **API Principal** | `GET /health`         | `{"status": "healthy", "redis": "connected", "pinecone": "connected"}` |
| **Hayhooks**      | `GET /status`         | `{"status": "ready", "pipelines": [...]}`                              |
| **OpenWebUI**     | `GET /health`         | `200 OK`                                                               |
| **Redis**         | `redis-cli ping`      | `PONG`                                                                 |
| **Streamlit**     | `GET /_stcore/health` | `200 OK`                                                               |

### Comandos de Diagnóstico Detalhado

```bash
# Status completo do sistema
curl http://localhost:8000/health | jq '.'

# Estatísticas do cache
curl http://localhost:8000/cache/stats | jq '.'

# Informações dos pipelines
curl http://localhost:1416/status | jq '.'

# Modelos disponíveis
curl http://localhost:8000/api/models | jq '.'

# Namespaces ativos
curl http://localhost:8000/api/namespaces | jq '.'

# Métricas de documentos
curl http://localhost:8000/api/documents | jq '.'
```

## 🚨 Problemas Comuns e Soluções

### 1. **Erro de API Key**

#### Sintomas:

```
HTTP 401: {"detail": "Invalid API key"}
Authentication failed for OpenAI API
```

#### Diagnóstico:

```bash
# Verificar variáveis de ambiente
docker-compose exec hayhooks env | grep -E "(OPENAI|PINECONE|ANTHROPIC)_API_KEY"

# Testar API key diretamente
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

#### Soluções:

1. **Verificar .env file**:

   ```bash
   cat .env | grep API_KEY
   ```

2. **Recarregar configurações**:

   ```bash
   docker-compose down
   docker-compose up -d
   ```

3. **Validar formato da API key**:
   - OpenAI: `sk-...` (51 chars)
   - Pinecone: UUID format
   - Anthropic: `sk-ant-...`

### 2. **Pinecone Connection Issues**

#### Sintomas:

```
PineconeApiException: Index 'haystack-docs' not found
Connection timeout to Pinecone
```

#### Diagnóstico:

```bash
# Testar conexão Pinecone
curl http://localhost:8000/rag/info

# Verificar configuração
docker-compose exec hayhooks env | grep PINECONE
```

#### Soluções:

1. **Verificar configuração**:

   ```bash
   # Verificar se o índice existe
   python3 -c "
   import pinecone
   pinecone.init(api_key='$PINECONE_API_KEY', environment='$PINECONE_ENVIRONMENT')
   print(pinecone.list_indexes())
   "
   ```

2. **Criar índice se necessário**:

   ```python
   import pinecone
   pinecone.init(api_key="your-api-key", environment="your-env")

   # Criar índice
   pinecone.create_index(
       name="haystack-docs",
       dimension=1536,  # OpenAI embeddings
       metric="cosine"
   )
   ```

3. **Verificar região/ambiente**:
   ```bash
   # Ambientes comuns: gcp-starter, us-east-1-aws, etc.
   echo $PINECONE_ENVIRONMENT
   ```

### 3. **Redis Cache Problems**

#### Sintomas:

```
redis.exceptions.ConnectionError: Error connecting to Redis
Cache miss for all queries
Slow response times
```

#### Diagnóstico:

```bash
# Testar conexão Redis
docker-compose exec redis redis-cli ping

# Verificar uso de memória
docker-compose exec redis redis-cli info memory

# Verificar estatísticas
curl http://localhost:8000/cache/stats
```

#### Soluções:

1. **Restart Redis**:

   ```bash
   docker-compose restart redis
   ```

2. **Limpar cache**:

   ```bash
   docker-compose exec redis redis-cli flushall
   ```

3. **Verificar configuração de memória**:
   ```bash
   # Aumentar memória se necessário
   docker-compose exec redis redis-cli config set maxmemory 256mb
   ```

### 4. **OpenWebUI não carrega**

#### Sintomas:

```
HTTP 500: Internal Server Error
Connection refused on port 3000
Login page não aparece
```

#### Diagnóstico:

```bash
# Verificar logs
docker-compose logs open-webui

# Verificar processo
docker-compose exec open-webui ps aux

# Testar endpoint
curl http://localhost:3000/health
```

#### Soluções:

1. **Restart serviço**:

   ```bash
   docker-compose restart open-webui
   ```

2. **Rebuild se necessário**:

   ```bash
   docker-compose build open-webui
   docker-compose up -d open-webui
   ```

3. **Verificar variáveis de ambiente**:
   ```bash
   docker-compose exec open-webui env | grep -E "(SECRET_KEY|WEBUI)"
   ```

### 5. **Upload de Documentos Falha**

#### Sintomas:

```
HTTP 413: Request Entity Too Large
Upload processing timeout
Files not appearing in vector store
```

#### Diagnóstico:

```bash
# Verificar logs do hayhooks
docker-compose logs hayhooks | grep -i upload

# Testar upload simples
curl -X POST "http://localhost:8000/upload" \
  -F "file=@small-test.txt"

# Verificar namespaces
curl http://localhost:8000/api/namespaces
```

#### Soluções:

1. **Aumentar limites de upload**:

   ```yaml
   # Em docker-compose.yml
   hayhooks:
     environment:
       - MAX_UPLOAD_SIZE=100MB
   ```

2. **Verificar formatos suportados**:

   ```bash
   # Formatos: PDF, TXT, DOCX, MD, CSV, JSON
   file your-document.pdf
   ```

3. **Upload por partes**:
   ```bash
   # Dividir documentos grandes
   split -l 1000 large-file.txt chunk_
   ```

### 6. **Performance Issues**

#### Sintomas:

```
Query response > 10 seconds
High CPU usage
Memory exhaustion
```

#### Diagnóstico:

```bash
# Monitorar recursos
docker stats

# Verificar cache hit rate
curl http://localhost:8000/cache/stats | jq '.hit_rate'

# Verificar embedding cache
curl http://localhost:8000/api/embeddings/stats
```

#### Soluções:

1. **Otimizar cache**:

   ```bash
   # Aumentar TTL do cache
   # Configurar em hayhooks/cache_manager.py
   CACHE_TTL = 3600  # 1 hora
   ```

2. **Reduzir chunk size**:

   ```python
   # Em pipelines/rag_pipeline.py
   chunk_size = 500  # Reduzir de 1000
   chunk_overlap = 100  # Reduzir overlap
   ```

3. **Connection pooling**:
   ```python
   # Configurar pool de conexões
   redis_pool = redis.ConnectionPool(max_connections=20)
   ```

## 📊 Análise de Logs

### Estrutura de Logs

```bash
# Logs em tempo real
docker-compose logs -f

# Logs específicos por serviço
docker-compose logs hayhooks      # Backend principal
docker-compose logs open-webui    # Interface web
docker-compose logs redis         # Cache
docker-compose logs streamlit-upload  # Upload frontend
```

### Patterns de Log Importantes

#### 1. **Logs de Sucesso**

```
INFO: Application startup complete
INFO: Document uploaded successfully: doc_id=abc123
INFO: RAG query processed in 1.2s
INFO: Cache hit for query: query_hash=def456
```

#### 2. **Logs de Erro**

```
ERROR: Failed to connect to Pinecone: [details]
ERROR: OpenAI API rate limit exceeded
ERROR: Redis connection timeout
WARNING: Large document size: 15MB
```

#### 3. **Performance Logs**

```
INFO: Query embedding took 0.2s
INFO: Vector search took 0.8s
INFO: LLM generation took 2.1s
INFO: Total query time: 3.1s
```

### Log Analysis Scripts

```bash
#!/bin/bash
# analyze-logs.sh

LOG_FILE="logs/hayhooks.log"

echo "📊 Log Analysis Report"
echo "======================"

# Error rate
echo "🚨 Error Rate:"
ERROR_COUNT=$(grep -c "ERROR" $LOG_FILE)
TOTAL_LINES=$(wc -l < $LOG_FILE)
echo "   Errors: $ERROR_COUNT / $TOTAL_LINES lines"

# Performance metrics
echo "⏱️ Average Response Times:"
grep "Total query time" $LOG_FILE | awk '{print $NF}' | \
  sed 's/s$//' | awk '{sum+=$1; count++} END {print "   Query: " sum/count "s"}'

# Most common errors
echo "🔍 Top Errors:"
grep "ERROR" $LOG_FILE | cut -d':' -f3- | sort | uniq -c | sort -nr | head -5

# Cache performance
echo "💾 Cache Performance:"
grep "Cache hit" $LOG_FILE | wc -l | awk '{print "   Cache hits: " $1}'
grep "Cache miss" $LOG_FILE | wc -l | awk '{print "   Cache misses: " $1}'
```

## 📈 Monitoramento e Performance

### Métricas Chave

#### 1. **Response Times**

```bash
# Monitorar latência das APIs
while true; do
  time curl -s http://localhost:8000/health >/dev/null
  sleep 5
done
```

#### 2. **Resource Usage**

```bash
# CPU e Memória
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Disk Usage
du -sh data/
df -h
```

#### 3. **Cache Metrics**

```bash
# Redis stats
docker-compose exec redis redis-cli info stats | grep -E "(hits|misses|hit_rate)"

# Application cache
curl http://localhost:8000/cache/stats | jq '{hit_rate, total_queries, cache_size}'
```

### Performance Monitoring Script

```bash
#!/bin/bash
# monitor-performance.sh

INTERVAL=60  # segundos
LOG_FILE="performance.log"

echo "Starting performance monitoring (interval: ${INTERVAL}s)"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # API Response time
    API_TIME=$(time (curl -s http://localhost:8000/health >/dev/null) 2>&1 | \
              grep real | awk '{print $2}')

    # Memory usage
    MEM_USAGE=$(docker stats --no-stream --format "{{.MemUsage}}" hayhooks | \
               head -1)

    # Cache hit rate
    HIT_RATE=$(curl -s http://localhost:8000/cache/stats | jq -r '.hit_rate // 0')

    # Log metrics
    echo "$TIMESTAMP,API:$API_TIME,MEM:$MEM_USAGE,CACHE:$HIT_RATE" >> $LOG_FILE

    sleep $INTERVAL
done
```

### Alertas e Thresholds

```bash
#!/bin/bash
# check-thresholds.sh

# Thresholds
MAX_RESPONSE_TIME=5.0    # segundos
MIN_CACHE_HIT_RATE=0.6   # 60%
MAX_MEMORY_MB=2048       # 2GB

# Check API response time
RESPONSE_TIME=$(curl -w "%{time_total}" -s http://localhost:8000/health >/dev/null)
if (( $(echo "$RESPONSE_TIME > $MAX_RESPONSE_TIME" | bc -l) )); then
    echo "🚨 ALERT: API response time too high: ${RESPONSE_TIME}s"
fi

# Check cache hit rate
HIT_RATE=$(curl -s http://localhost:8000/cache/stats | jq -r '.hit_rate // 0')
if (( $(echo "$HIT_RATE < $MIN_CACHE_HIT_RATE" | bc -l) )); then
    echo "🚨 ALERT: Cache hit rate too low: ${HIT_RATE}"
fi

# Check memory usage
MEM_MB=$(docker stats --no-stream --format "{{.MemUsage}}" hayhooks | \
         sed 's/MiB.*//' | sed 's/.* //')
if (( $(echo "$MEM_MB > $MAX_MEMORY_MB" | bc -l) )); then
    echo "🚨 ALERT: Memory usage too high: ${MEM_MB}MB"
fi
```

## ⚡ Otimização de Sistema

### 1. **Cache Optimization**

```python
# cache_manager.py optimizations
class OptimizedCacheManager:
    def __init__(self):
        self.redis_pool = redis.ConnectionPool(
            max_connections=20,
            retry_on_timeout=True,
            socket_timeout=5
        )

        # Multi-level TTL
        self.ttl_config = {
            "embeddings": 86400,      # 24 horas
            "query_results": 3600,    # 1 hora
            "document_meta": 43200,   # 12 horas
        }

    async def smart_cache_key(self, content: str) -> str:
        """Generate semantic cache key"""
        # Use content hash + embeddings similarity
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"smart_cache:{content_hash}"
```

### 2. **Database Optimization**

```python
# Pinecone optimization
class OptimizedPinecone:
    def __init__(self):
        self.batch_size = 100
        self.connection_pool = ConnectionPool(max_connections=10)

    async def batch_upsert(self, vectors: List[Dict]):
        """Batch upload for better performance"""
        for i in range(0, len(vectors), self.batch_size):
            batch = vectors[i:i + self.batch_size]
            await self.index.upsert(vectors=batch)
            await asyncio.sleep(0.1)  # Rate limiting
```

### 3. **Resource Management**

```yaml
# docker-compose.yml resource limits
services:
  hayhooks:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
        reservations:
          memory: 1G
          cpus: "0.5"

  redis:
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: "0.5"
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
```

### 4. **Query Optimization**

```python
# Optimized RAG pipeline
class OptimizedRAGPipeline:
    def __init__(self):
        # Reduced embedding dimensions for faster search
        self.embedding_model = "text-embedding-3-small"  # 1536 dims vs 3072

        # Optimized retrieval
        self.top_k = 5  # Reduced from 10
        self.score_threshold = 0.7  # Higher threshold

        # Async processing
        self.enable_concurrent_embedding = True

    async def optimized_query(self, query: str):
        # Parallel processing
        tasks = [
            self.embed_query(query),
            self.check_cache(query),
            self.warm_up_llm()
        ]
        results = await asyncio.gather(*tasks)
        return await self.process_results(results)
```

## 🛠️ Scripts de Diagnóstico

### Script de Diagnóstico Completo

```bash
#!/bin/bash
# comprehensive-diagnostic.sh

echo "🔍 HAYSTACK RAG SYSTEM - COMPREHENSIVE DIAGNOSTIC"
echo "=================================================="
date
echo ""

# Function to check service
check_service() {
    local name=$1
    local url=$2
    local expected=$3

    echo -n "Checking $name... "
    response=$(curl -s -w "%{http_code}" -o /dev/null "$url" 2>/dev/null)

    if [ "$response" = "$expected" ]; then
        echo "✅ OK"
    else
        echo "❌ FAIL (HTTP $response)"
    fi
}

# 1. DOCKER ENVIRONMENT
echo "🐳 DOCKER ENVIRONMENT"
echo "---------------------"
docker --version
docker-compose --version
echo ""

# 2. CONTAINER STATUS
echo "📦 CONTAINER STATUS"
echo "-------------------"
docker-compose ps
echo ""

# 3. SERVICE ENDPOINTS
echo "🔌 SERVICE ENDPOINTS"
echo "--------------------"
check_service "Hayhooks API" "http://localhost:8000/health" "200"
check_service "Hayhooks Pipelines" "http://localhost:1416/status" "200"
check_service "OpenWebUI" "http://localhost:3000" "200"
check_service "Streamlit Upload" "http://localhost:8501/_stcore/health" "200"
echo ""

# 4. DATABASE CONNECTIONS
echo "🗄️ DATABASE CONNECTIONS"
echo "------------------------"
echo -n "Redis... "
if docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
    echo "✅ Connected"
else
    echo "❌ Failed"
fi
echo ""

# 5. EXTERNAL APIS
echo "🌐 EXTERNAL API KEYS"
echo "--------------------"
if [ -n "$OPENAI_API_KEY" ]; then
    echo -n "OpenAI API... "
    if curl -s -H "Authorization: Bearer $OPENAI_API_KEY" \
       https://api.openai.com/v1/models >/dev/null 2>&1; then
        echo "✅ Valid"
    else
        echo "❌ Invalid"
    fi
else
    echo "⚠️  OpenAI API Key not set"
fi

if [ -n "$PINECONE_API_KEY" ]; then
    echo "✅ Pinecone API Key set"
else
    echo "⚠️  Pinecone API Key not set"
fi
echo ""

# 6. SYSTEM RESOURCES
echo "💻 SYSTEM RESOURCES"
echo "-------------------"
echo "Memory usage:"
docker stats --no-stream --format "  {{.Container}}: {{.MemUsage}}"
echo ""
echo "Disk usage:"
df -h / | tail -1 | awk '{print "  Root: " $3 " used / " $2 " total (" $5 " full)"}'
echo ""

# 7. LOG ANALYSIS
echo "📋 RECENT ERRORS"
echo "----------------"
echo "Last 10 errors from logs:"
docker-compose logs --tail=100 | grep -i error | tail -10 | \
  sed 's/^/  /' || echo "  No recent errors found"
echo ""

# 8. PERFORMANCE METRICS
echo "📊 PERFORMANCE METRICS"
echo "----------------------"
if curl -s http://localhost:8000/cache/stats >/dev/null 2>&1; then
    echo "Cache statistics:"
    curl -s http://localhost:8000/cache/stats | jq -r '
      "  Hit rate: " + (.hit_rate // 0 | tostring) + "%",
      "  Total queries: " + (.total_queries // 0 | tostring),
      "  Cache size: " + (.cache_size // "unknown" | tostring)
    ' 2>/dev/null || echo "  Cache stats unavailable"
else
    echo "  Cache statistics unavailable"
fi
echo ""

# 9. RECOMMENDATIONS
echo "💡 RECOMMENDATIONS"
echo "------------------"

# Check for common issues
if ! docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
    echo "  - Fix Redis connection"
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "  - Set OPENAI_API_KEY in .env file"
fi

if [ -z "$PINECONE_API_KEY" ]; then
    echo "  - Set PINECONE_API_KEY in .env file"
fi

# Check response times
response_time=$(curl -w "%{time_total}" -s http://localhost:8000/health >/dev/null 2>&1)
if (( $(echo "$response_time > 2.0" | bc -l) 2>/dev/null )); then
    echo "  - API response time is slow (${response_time}s), consider optimization"
fi

echo ""
echo "📅 Diagnostic completed at $(date)"
echo "For detailed troubleshooting, see: docs/TROUBLESHOOTING.md"
```

### Performance Benchmark Script

```bash
#!/bin/bash
# benchmark-performance.sh

echo "🚀 PERFORMANCE BENCHMARK"
echo "========================"

# Test upload performance
echo "📤 Testing upload performance..."
time curl -X POST "http://localhost:8000/upload" \
  -F "file=@test-documents/sample.pdf" \
  -F "namespace=benchmark" >/dev/null 2>&1

# Test query performance
echo "🔍 Testing query performance..."
for i in {1..10}; do
    time curl -X POST "http://localhost:8000/rag/query" \
      -H "Content-Type: application/json" \
      -d '{"query": "What is this document about?", "namespace": "benchmark"}' \
      >/dev/null 2>&1
done

# Test cache performance
echo "💾 Testing cache performance..."
# First query (cache miss)
time curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Cache test query", "namespace": "benchmark"}' \
  >/dev/null 2>&1

# Second query (cache hit)
time curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Cache test query", "namespace": "benchmark"}' \
  >/dev/null 2>&1

echo "Benchmark completed!"
```

---

## 🚨 Emergency Procedures

### Quick Reset

```bash
#!/bin/bash
# emergency-reset.sh

echo "🚨 EMERGENCY SYSTEM RESET"
echo "========================="

# Stop all services
docker-compose down

# Clear cache
docker-compose exec redis redis-cli flushall 2>/dev/null || true

# Clear logs
docker-compose logs > backup-logs-$(date +%Y%m%d-%H%M%S).log
docker system prune -f

# Restart services
docker-compose up -d

# Wait for services
sleep 30

# Health check
curl http://localhost:8000/health
```

### Backup Critical Data

```bash
#!/bin/bash
# backup-system.sh

BACKUP_DIR="backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup Redis data
docker-compose exec redis redis-cli save
docker cp $(docker-compose ps -q redis):/data/dump.rdb $BACKUP_DIR/

# Backup configuration
cp .env $BACKUP_DIR/
cp docker-compose.yml $BACKUP_DIR/

# Backup logs
docker-compose logs > $BACKUP_DIR/logs.txt

echo "Backup created in $BACKUP_DIR"
```

---

**📅 Última Atualização**: Janeiro 2024  
**🔄 Próxima Revisão**: Mensal  
**👥 Responsável**: DevOps + Tech Lead

Para **issues específicas** não cobertas neste guia, consulte:

- 📖 [README.md](../README.md) para overview
- 🚀 [DEPLOYMENT.md](DEPLOYMENT.md) para configuração
- 🔌 [API.md](API.md) para APIs
- 🏗️ [ARCHITECTURE.md](ARCHITECTURE.md) para arquitetura
