# ⚡ Quick Start Guide - Haystack RAG System

Este guia permite que você tenha o sistema funcionando em **menos de 5 minutos** com as configurações mais comuns.

> 📖 **Para visão geral completa**: [README.md](../README.md)  
> 🚀 **Para deploy avançado**: [DEPLOYMENT.md](DEPLOYMENT.md)  
> 🔌 **Para integração**: [API.md](API.md)

## 🎯 Cenários de Deploy Rápido

### 🏠 Local Development (Recomendado para testes)

```bash
# 1. Clone e configure
git clone <repository-url>
cd haystack
cp local.env.example .env

# 2. Configure API keys mínimas
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
echo "PINECONE_API_KEY=your-pinecone-key" >> .env
echo "PINECONE_INDEX_NAME=haystack-docs" >> .env
echo "PINECONE_ENVIRONMENT=gcp-starter" >> .env

# 3. Deploy
docker-compose up -d

# 4. Aguarde (30-60s) e acesse
echo "✅ OpenWebUI: http://localhost:3000"
echo "✅ Upload: http://localhost:8501"
echo "✅ API: http://localhost:8000/docs"
```

### 🔧 Development Mode (Com hot-reload)

```bash
# Após setup básico acima
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Logs em tempo real
docker-compose logs -f hayhooks
```

### 🏭 Production Quick Deploy

```bash
# 1. Setup básico
git clone <repository-url> && cd haystack
cp local.env.example .env

# 2. Configure para produção
cat >> .env << EOF
WEBUI_AUTH=true
ENABLE_SIGNUP=false
DEBUG=false
REDIS_PASSWORD=your-secure-password
WEBUI_SECRET_KEY=your-secret-key
EOF

# 3. Deploy com proxy
docker-compose --profile proxy up -d
```

## 🔑 Configuração Mínima de API Keys

### Obrigatórias

```bash
# OpenAI (LLM + Embeddings)
OPENAI_API_KEY=sk-proj-...

# Pinecone (Vector Database)
PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
PINECONE_INDEX_NAME=haystack-docs
PINECONE_ENVIRONMENT=gcp-starter
```

### Opcionais

```bash
# Anthropic (Claude models)
ANTHROPIC_API_KEY=sk-ant-...

# Redis (Cache - auto-gerado se não definido)
REDIS_PASSWORD=redis123
```

## 🚀 Verificação Rápida

### 1. Status dos Serviços

```bash
# Verificar se todos estão rodando
docker-compose ps

# Deve mostrar:
# ✅ hayhooks (healthy)
# ✅ open-webui (healthy)
# ✅ redis (healthy)
# ✅ streamlit-upload (healthy)
```

### 2. Health Check

```bash
# API principal
curl http://localhost:8000/health

# Deve retornar: {"status": "healthy", ...}
```

### 3. Teste de Upload

```bash
# Upload de teste
curl -X POST "http://localhost:8000/upload" \
  -F "file=@README.md" \
  -F "namespace=test"

# Deve retornar: {"success": true, ...}
```

### 4. Teste de RAG

```bash
# Consulta de teste
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this system about?"}'

# Deve retornar resposta com base no README.md
```

## 🎯 Primeiros Passos

### 1. Acesse a Interface Principal

```
http://localhost:3000
```

- Crie uma conta (se `ENABLE_SIGNUP=true`)
- Selecione o modelo `haystack-rag`
- Comece a fazer perguntas

### 2. Faça Upload de Documentos

```
http://localhost:8501
```

- Arraste arquivos PDF, TXT, DOCX
- Configure namespace (ex: "meus-docs")
- Clique em "Process Documents"

### 3. Teste a API

```
http://localhost:8000/docs
```

- Documentação interativa Swagger
- Teste endpoints diretamente
- Veja exemplos de código

## ⚡ Troubleshooting Express

| Problema                  | Comando de Diagnóstico                          | Solução                        |
| ------------------------- | ----------------------------------------------- | ------------------------------ |
| **Serviços não sobem**    | `docker-compose ps`                             | `docker-compose up -d`         |
| **API Key inválida**      | `docker-compose logs hayhooks \| grep -i error` | Verificar .env                 |
| **Pinecone erro**         | `curl http://localhost:8000/rag/info`           | Verificar índice               |
| **Redis não conecta**     | `docker-compose exec redis redis-cli ping`      | `docker-compose restart redis` |
| **OpenWebUI não carrega** | `docker-compose logs open-webui`                | Aguardar 60s                   |
| **Upload falha**          | `curl -I http://localhost:8501`                 | Verificar Streamlit            |

### Comandos de Reset Rápido

```bash
# Reset completo (CUIDADO: apaga dados)
docker-compose down -v
docker-compose up -d

# Reset apenas cache
docker-compose exec redis redis-cli flushall

# Rebuild após mudanças
docker-compose build --no-cache
docker-compose up -d
```

## 📊 Monitoramento Básico

### URLs Importantes

| Serviço             | URL                               | Descrição            |
| ------------------- | --------------------------------- | -------------------- |
| **Health Check**    | http://localhost:8000/health      | Status geral         |
| **API Docs**        | http://localhost:8000/docs        | Documentação Swagger |
| **Cache Stats**     | http://localhost:8000/cache/stats | Estatísticas Redis   |
| **Pipeline Status** | http://localhost:1416/status      | Status Haystack      |

### Logs Úteis

```bash
# Logs principais
docker-compose logs hayhooks

# Logs de erro
docker-compose logs hayhooks | grep -i error

# Logs em tempo real
docker-compose logs -f hayhooks open-webui
```

## 🔄 Próximos Passos

Após o sistema estar funcionando:

1. **📚 Leia a documentação completa**: [README.md](../README.md)
2. **🚀 Configure para produção**: [DEPLOYMENT.md](DEPLOYMENT.md)
3. **🔌 Integre com sua aplicação**: [API.md](API.md)
4. **🏗️ Entenda a arquitetura**: [ARCHITECTURE.md](ARCHITECTURE.md)

## 🆘 Suporte

Se algo não funcionar:

1. **Verifique os logs**: `docker-compose logs`
2. **Consulte troubleshooting**: [DEPLOYMENT.md#troubleshooting](DEPLOYMENT.md#troubleshooting)
3. **Teste conectividade**: Scripts de verificação no DEPLOYMENT.md

---

**⏱️ Tempo estimado**: 5 minutos  
**💾 Espaço necessário**: ~2GB  
**🔧 Pré-requisitos**: Docker + API keys
