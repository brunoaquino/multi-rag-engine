# 🚀 Guia de Deployment - Haystack RAG System

Este guia fornece instruções detalhadas para deploy do sistema Haystack RAG em diferentes ambientes, desde desenvolvimento local até produção.

> 📖 **Para uma visão geral do sistema e quick start básico**, consulte primeiro o [README.md](../README.md)
>
> 🔌 **Para documentação completa das APIs**, veja [API.md](API.md)

## 📋 Índice

- [Pré-requisitos](#-pré-requisitos)
- [Deployment Local](#-deployment-local)
- [Deployment de Desenvolvimento](#-deployment-de-desenvolvimento)
- [Deployment de Produção](#-deployment-de-produção)
- [Deployment na Nuvem](#-deployment-na-nuvem)
- [Configurações de Ambiente](#-configurações-de-ambiente)
- [Monitoramento](#-monitoramento)
- [Backup e Recuperação](#-backup-e-recuperação)
- [Troubleshooting](#-troubleshooting)

## 🔧 Pré-requisitos

### Requisitos Mínimos

| Componente         | Versão Mínima | Recomendado |
| ------------------ | ------------- | ----------- |
| **Docker**         | 20.10.0       | 24.0+       |
| **Docker Compose** | 2.0.0         | 2.20+       |
| **RAM**            | 4GB           | 8GB+        |
| **Armazenamento**  | 10GB          | 50GB+       |
| **CPU**            | 2 cores       | 4+ cores    |

### APIs Obrigatórias

- ✅ **Pinecone API Key** - Vector database
- ✅ **OpenAI API Key** - LLM e embeddings
- ⚠️ **Anthropic API Key** - Opcional (Claude models)

### Verificação de Pré-requisitos

```bash
# Verificar Docker
docker --version
docker-compose --version

# Verificar recursos do sistema
docker system info | grep -E "CPUs|Total Memory"

# Teste de conectividade com APIs
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

## 🏠 Deployment Local

### 1. Clone e Configuração Inicial

```bash
# Clone do repositório
git clone <repository-url>
cd haystack

# Criar arquivo de ambiente
cp local.env.example .env
```

### 2. Configurar Variáveis de Ambiente

Edite o arquivo `.env`:

```bash
# === API KEYS (Obrigatórias) ===
OPENAI_API_KEY=sk-your-openai-key-here
PINECONE_API_KEY=your-pinecone-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here  # Opcional

# === PINECONE CONFIGURATION ===
PINECONE_INDEX_NAME=haystack-docs
PINECONE_ENVIRONMENT=gcp-starter

# === SEGURANÇA ===
REDIS_PASSWORD=redis123
WEBUI_SECRET_KEY=your-secret-key-change-this
WEBUI_JWT_SECRET_KEY=your-jwt-secret-change-this

# === CONFIGURAÇÕES OPCIONAIS ===
WEBUI_AUTH=false              # Autenticação no OpenWebUI
ENABLE_SIGNUP=true            # Permitir novos usuários
DEFAULT_USER_ROLE=user        # Papel padrão dos usuários

# === PORTAS (Configuráveis) ===
WEBUI_PORT=3000               # OpenWebUI
STREAMLIT_PORT=8501           # Upload Frontend
HAYHOOKS_PORT=1416            # Hayhooks Pipelines
REDIS_PORT=6379               # Redis Cache
```

### 3. Inicializar o Sistema

```bash
# Build das imagens
docker-compose build

# Iniciar todos os serviços
docker-compose up -d

# Verificar status
docker-compose ps

# Aguardar inicialização completa
./scripts/health-check.sh
```

### 4. Verificação da Instalação

```bash
# Health check geral
curl http://localhost:8000/health

# Testar upload
curl -X POST "http://localhost:8000/upload" \
  -F "file=@README.md" \
  -F "namespace=test"

# Testar RAG
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this system about?"}'
```

## 🔧 Deployment de Desenvolvimento

### Configuração para Desenvolvimento

```bash
# Arquivo .env.development
cp .env .env.development

# Configurações específicas para dev
echo "
# === DESENVOLVIMENTO ===
DEBUG=true
LOG_LEVEL=debug
ENABLE_CORS=true
RELOAD_ON_CHANGE=true

# === CACHE DESABILITADO ===
ENABLE_CACHE=false
REDIS_TTL=60

# === VOLUMES DE DESENVOLVIMENTO ===
MOUNT_SOURCE_CODE=true
" >> .env.development
```

### Docker Compose Override

Crie `docker-compose.override.yml`:

```yaml
version: "3.8"

services:
  hayhooks:
    volumes:
      # Mount do código fonte para reload automático
      - ./hayhooks:/app:rw
      - ./scripts:/scripts:ro
    environment:
      - DEBUG=true
      - RELOAD_ON_CHANGE=true
    command: >
      uvicorn direct_api:app 
      --host 0.0.0.0 
      --port 8000 
      --reload 
      --log-level debug

  streamlit-upload:
    volumes:
      # Mount do código fonte
      - ./streamlit-upload:/app:rw
    command: >
      streamlit run app.py 
      --server.port=8501 
      --server.address=0.0.0.0 
      --server.runOnSave=true
      --logger.level=debug

  # Volumes de desenvolvimento
volumes:
  dev-logs:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./logs
```

### Executar em Modo Desenvolvimento

```bash
# Usar configuração de desenvolvimento
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Logs em tempo real
docker-compose logs -f hayhooks

# Debug de API específica
docker-compose exec hayhooks python -m debugpy --listen 0.0.0.0:5678 direct_api.py
```

## 🏭 Deployment de Produção

### 1. Configuração de Produção

```bash
# Arquivo .env.production
cp .env .env.production

# Configurações de produção
echo "
# === PRODUÇÃO ===
DEBUG=false
LOG_LEVEL=info
ENVIRONMENT=production

# === SEGURANÇA ===
WEBUI_AUTH=true
ENABLE_SIGNUP=false
DEFAULT_USER_ROLE=user

# === PERFORMANCE ===
ENABLE_CACHE=true
REDIS_TTL=3600
MAX_WORKERS=4

# === SSL/TLS ===
USE_SSL=true
SSL_CERT_PATH=/etc/ssl/certs/server.crt
SSL_KEY_PATH=/etc/ssl/private/server.key

# === BACKUP ===
BACKUP_ENABLED=true
BACKUP_SCHEDULE='0 2 * * *'  # Daily at 2 AM
" >> .env.production
```

### 2. Docker Compose para Produção

Crie `docker-compose.prod.yml`:

```yaml
version: "3.8"

services:
  # === REVERSE PROXY ===
  nginx:
    image: nginx:alpine
    container_name: nginx-prod
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.prod.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/ssl:ro
      - nginx-logs:/var/log/nginx
    restart: unless-stopped
    depends_on:
      - open-webui
      - hayhooks
    networks:
      - haystack-network

  # === APLICAÇÃO ===
  hayhooks:
    restart: unless-stopped
    environment:
      - DEBUG=false
      - LOG_LEVEL=info
      - MAX_WORKERS=4
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
        reservations:
          memory: 1G
          cpus: "0.5"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  open-webui:
    restart: unless-stopped
    environment:
      - WEBUI_AUTH=true
      - ENABLE_SIGNUP=false
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: "0.5"

  redis:
    restart: unless-stopped
    command: >
      redis-server 
      --appendonly yes 
      --appendfsync everysec
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: "0.5"

  # === BACKUP SERVICE ===
  backup:
    image: alpine:latest
    container_name: backup-service
    volumes:
      - redis-data:/backup/redis:ro
      - ./data:/backup/data:ro
      - ./backups:/backups:rw
    environment:
      - BACKUP_SCHEDULE=0 2 * * *
    command: >
      sh -c "
      apk add --no-cache dcron tar gzip &&
      echo '0 2 * * * /backup/scripts/backup.sh' | crontab - &&
      crond -f
      "
    restart: unless-stopped
    networks:
      - haystack-network

# === VOLUMES PERSISTENTES ===
volumes:
  redis-data-prod:
    name: redis-data-prod
  nginx-logs:
    name: nginx-logs

networks:
  haystack-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.21.0.0/16
```

### 3. Configuração Nginx para Produção

Crie `nginx/nginx.prod.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream webui {
        server open-webui:8080;
    }

    upstream api {
        server hayhooks:8000;
    }

    upstream upload {
        server streamlit-upload:8501;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=5r/s;

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL Configuration
        ssl_certificate /etc/ssl/certs/server.crt;
        ssl_certificate_key /etc/ssl/private/server.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;

        # Security Headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # Main application
        location / {
            proxy_pass http://webui;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Upload interface
        location /upload/ {
            limit_req zone=upload burst=10 nodelay;
            proxy_pass http://upload/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # File upload limits
            client_max_body_size 100M;
            proxy_request_buffering off;
        }

        # Health checks
        location /health {
            proxy_pass http://api/health;
            access_log off;
        }
    }
}
```

### 4. Deploy em Produção

```bash
# Build para produção
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# Deploy
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verificar deploy
./scripts/production-health-check.sh
```

## ☁️ Deployment na Nuvem

### AWS ECS

```bash
# Instalar AWS CLI
pip install awscli

# Configurar credenciais
aws configure

# Build e push para ECR
./scripts/deploy-aws.sh
```

### Google Cloud Run

```bash
# Instalar gcloud CLI
curl https://sdk.cloud.google.com | bash

# Autenticar
gcloud auth login

# Deploy
./scripts/deploy-gcp.sh
```

### Azure Container Instances

```bash
# Instalar Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Deploy
./scripts/deploy-azure.sh
```

### Kubernetes (K8s)

Crie arquivos de manifesto em `k8s/`:

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: haystack-rag

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: haystack-config
  namespace: haystack-rag
data:
  PINECONE_INDEX_NAME: "haystack-docs"
  PINECONE_ENVIRONMENT: "gcp-starter"
  # ... outras configurações não-sensíveis

---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: haystack-secrets
  namespace: haystack-rag
type: Opaque
data:
  OPENAI_API_KEY: # base64 encoded
  PINECONE_API_KEY: # base64 encoded
  ANTHROPIC_API_KEY: # base64 encoded

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hayhooks
  namespace: haystack-rag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hayhooks
  template:
    metadata:
      labels:
        app: hayhooks
    spec:
      containers:
        - name: hayhooks
          image: your-registry/hayhooks:latest
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: haystack-config
            - secretRef:
                name: haystack-secrets
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
```

Deploy no Kubernetes:

```bash
# Aplicar manifestos
kubectl apply -f k8s/

# Verificar status
kubectl get pods -n haystack-rag

# Logs
kubectl logs -f deployment/hayhooks -n haystack-rag
```

## 🔧 Configurações de Ambiente

### Variáveis de Ambiente Críticas

| Variável               | Obrigatória | Descrição           | Exemplo         |
| ---------------------- | ----------- | ------------------- | --------------- |
| `OPENAI_API_KEY`       | ✅          | API key da OpenAI   | `sk-...`        |
| `PINECONE_API_KEY`     | ✅          | API key do Pinecone | `xxx-xxx-xxx`   |
| `PINECONE_INDEX_NAME`  | ✅          | Nome do índice      | `haystack-docs` |
| `PINECONE_ENVIRONMENT` | ✅          | Ambiente Pinecone   | `gcp-starter`   |
| `REDIS_PASSWORD`       | ⚠️          | Senha do Redis      | `redis123`      |
| `WEBUI_SECRET_KEY`     | ⚠️          | Chave secreta WebUI | `random-string` |

### Configurações de Performance

```bash
# Cache Configuration
ENABLE_CACHE=true
REDIS_TTL=3600
REDIS_MAX_CONNECTIONS=100

# Processing Configuration
MAX_WORKERS=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE=50

# Model Configuration
DEFAULT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
MAX_TOKENS=4000
TEMPERATURE=0.1
```

### Configurações de Segurança

```bash
# Authentication
WEBUI_AUTH=true
ENABLE_SIGNUP=false
DEFAULT_USER_ROLE=user
SESSION_TIMEOUT=3600

# CORS & Security
ENABLE_CORS=false
ALLOWED_ORIGINS=https://your-domain.com
RATE_LIMIT=100/minute
```

## 📊 Monitoramento

### Health Checks Automáticos

```bash
# Script de monitoramento
#!/bin/bash
# scripts/monitor.sh

while true; do
    # Check API health
    if ! curl -sf http://localhost:8000/health > /dev/null; then
        echo "$(date): API health check failed" >> /var/log/haystack/monitor.log
        # Send alert
    fi

    # Check Redis
    if ! docker-compose exec redis redis-cli ping > /dev/null; then
        echo "$(date): Redis health check failed" >> /var/log/haystack/monitor.log
    fi

    # Check disk space
    DISK_USAGE=$(df /var/lib/docker | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ $DISK_USAGE -gt 80 ]; then
        echo "$(date): Disk usage high: ${DISK_USAGE}%" >> /var/log/haystack/monitor.log
    fi

    sleep 60
done
```

### Métricas com Prometheus

```yaml
# docker-compose.monitoring.yml
version: "3.8"

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--web.console.libraries=/etc/prometheus/console_libraries"
      - "--web.console.templates=/etc/prometheus/consoles"
    networks:
      - haystack-network

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3001:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    networks:
      - haystack-network

volumes:
  prometheus-data:
  grafana-data:
```

### Alertas

```bash
# Configurar alertas via email
echo "
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@your-domain.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'admin@your-domain.com'
    subject: 'Haystack RAG Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
" > monitoring/alertmanager.yml
```

## 💾 Backup e Recuperação

### Script de Backup Automático

```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="haystack_backup_${DATE}"

echo "Starting backup: $BACKUP_NAME"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Backup Redis data
docker-compose exec redis redis-cli --rdb /tmp/dump.rdb
docker-compose cp redis:/tmp/dump.rdb "${BACKUP_DIR}/${BACKUP_NAME}/redis.rdb"

# Backup application data
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/data.tar.gz" ./data/

# Backup configuration
cp .env "${BACKUP_DIR}/${BACKUP_NAME}/"
cp docker-compose*.yml "${BACKUP_DIR}/${BACKUP_NAME}/"

# Create backup info
echo "{
  \"backup_date\": \"$(date -Iseconds)\",
  \"version\": \"$(git rev-parse HEAD)\",
  \"components\": [\"redis\", \"data\", \"config\"]
}" > "${BACKUP_DIR}/${BACKUP_NAME}/backup_info.json"

# Compress final backup
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}/"
rm -rf "${BACKUP_NAME}/"

echo "Backup completed: ${BACKUP_NAME}.tar.gz"

# Cleanup old backups (keep last 7 days)
find "${BACKUP_DIR}" -name "haystack_backup_*.tar.gz" -mtime +7 -delete
```

### Restauração

```bash
#!/bin/bash
# scripts/restore.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

echo "Restoring from: $BACKUP_FILE"

# Stop services
docker-compose down

# Extract backup
TEMP_DIR=$(mktemp -d)
tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"
BACKUP_DIR=$(ls "$TEMP_DIR")

# Restore Redis data
docker-compose up -d redis
sleep 10
docker-compose cp "${TEMP_DIR}/${BACKUP_DIR}/redis.rdb" redis:/tmp/dump.rdb
docker-compose exec redis redis-cli DEBUG LOADAOF

# Restore application data
rm -rf ./data/
tar -xzf "${TEMP_DIR}/${BACKUP_DIR}/data.tar.gz"

# Restore configuration
cp "${TEMP_DIR}/${BACKUP_DIR}/.env" .

# Restart services
docker-compose up -d

echo "Restore completed"
```

## 🚨 Troubleshooting

### Problemas Comuns

#### 1. API Keys Inválidas

```bash
# Verificar API keys
docker-compose exec hayhooks env | grep API_KEY

# Testar conectividade
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models | jq '.data[0].id'
```

#### 2. Pinecone Connection Issues

```bash
# Verificar configuração Pinecone
curl "http://localhost:8000/rag/info" | jq

# Verificar índice
python3 -c "
import pinecone
pinecone.init(api_key='$PINECONE_API_KEY', environment='$PINECONE_ENVIRONMENT')
print(pinecone.list_indexes())
"
```

#### 3. Redis Connection Problems

```bash
# Verificar Redis
docker-compose exec redis redis-cli ping

# Verificar logs
docker-compose logs redis

# Reset Redis
docker-compose restart redis
```

#### 4. OpenWebUI Não Carrega

```bash
# Verificar logs
docker-compose logs open-webui

# Verificar conectividade
curl -I http://localhost:3000

# Reset completo
docker-compose restart open-webui
```

#### 5. Upload Errors

```bash
# Verificar limites de arquivo
curl -X POST "http://localhost:8000/upload" \
  -F "file=@large_file.pdf" \
  -v

# Verificar espaço em disco
df -h

# Limpar uploads antigos
docker-compose exec hayhooks find /tmp -name "*.pdf" -mtime +1 -delete
```

### Logs de Diagnóstico

```bash
# Coletar todos os logs
mkdir -p diagnosis/
docker-compose logs > diagnosis/docker-compose.log
docker-compose ps > diagnosis/services-status.txt
docker system df > diagnosis/docker-space.txt
curl http://localhost:8000/health > diagnosis/health-check.json

# Criar arquivo de diagnóstico
tar -czf diagnosis_$(date +%Y%m%d_%H%M%S).tar.gz diagnosis/
```

### Performance Issues

```bash
# Verificar uso de recursos
docker stats

# Verificar cache hit rate
curl http://localhost:8000/cache/stats | jq

# Otimizar Redis
docker-compose exec redis redis-cli memory doctor

# Limpar cache se necessário
docker-compose exec redis redis-cli flushall
```

---

## 📞 Suporte

Para suporte adicional:

1. Verifique os [logs de diagnóstico](#logs-de-diagnóstico)
2. Consulte a [documentação da API](API.md)
3. Verifique [issues no GitHub](issues-url)
4. Contate o suporte técnico

---

**Versão do Documento**: 1.0  
**Última Atualização**: $(date)  
**Compatibilidade**: Haystack RAG System v2.0+
