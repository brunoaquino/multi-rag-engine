# 🔌 API Documentation - Haystack RAG System

Este documento fornece documentação completa para todas as APIs disponíveis no sistema Haystack RAG, incluindo endpoints, parâmetros, exemplos de uso e códigos de resposta.

> 📖 **Para visão geral e quick start**, veja [README.md](../README.md)
>
> 🚀 **Para instruções de deploy e configuração**, consulte [DEPLOYMENT.md](DEPLOYMENT.md)
>
> 🏗️ **Para arquitetura do sistema**, veja [ARCHITECTURE.md](ARCHITECTURE.md)

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Autenticação](#-autenticação)
- [API Principal (Porta 8000)](#-api-principal-porta-8000)
- [API Haystack Nativa (Porta 1416)](#-api-haystack-nativa-porta-1416)
- [Códigos de Resposta](#-códigos-de-resposta)
- [Modelos de Dados](#-modelos-de-dados)
- [Exemplos de Integração](#-exemplos-de-integração)
- [Rate Limiting](#-rate-limiting)
- [Troubleshooting](#-troubleshooting)

## 🌟 Visão Geral

O sistema Haystack RAG expõe duas APIs principais:

| API               | Base URL                | Descrição               | Uso Principal              |
| ----------------- | ----------------------- | ----------------------- | -------------------------- |
| **API Principal** | `http://localhost:8000` | API customizada FastAPI | Upload, RAG queries, chat  |
| **API Haystack**  | `http://localhost:1416` | API nativa Hayhooks     | Pipelines Haystack diretos |

### Características das APIs

- **🔄 REST**: Arquitetura RESTful com JSON
- **📊 OpenAPI**: Documentação Swagger automática
- **⚡ Assíncrono**: Suporte a processamento assíncrono
- **🔒 Seguro**: Headers de segurança e rate limiting
- **📝 Logs**: Logging estruturado para debugging

### URLs de Documentação Interativa

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## 🔐 Autenticação

### API Keys

As APIs utilizam diferentes métodos de autenticação dependendo do endpoint:

```bash
# Configuração via variáveis de ambiente
OPENAI_API_KEY=sk-your-openai-key
PINECONE_API_KEY=your-pinecone-key
ANTHROPIC_API_KEY=your-anthropic-key  # Opcional
```

### Headers de Segurança

```http
Content-Type: application/json
Accept: application/json
User-Agent: Haystack-RAG-Client/1.0
X-Request-ID: unique-request-id  # Opcional para debugging
```

## 🚀 API Principal (Porta 8000)

### 🏥 Health Check

#### `GET /health`

Verifica o status geral do sistema e conectividade com serviços externos.

**Parâmetros**: Nenhum

**Resposta de Sucesso** (200):

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.0.0",
  "services": {
    "redis": {
      "status": "connected",
      "latency_ms": 2.3
    },
    "pinecone": {
      "status": "connected",
      "index": "haystack-docs",
      "dimensions": 1536
    },
    "openai": {
      "status": "connected",
      "models_available": ["gpt-4o", "gpt-4o-mini"]
    }
  },
  "system": {
    "memory_usage_mb": 1024,
    "disk_usage_percent": 45,
    "uptime_seconds": 86400
  }
}
```

**Exemplo de Uso**:

```bash
curl http://localhost:8000/health
```

### 📤 Upload de Documentos

#### `POST /upload`

Faz upload e processa documentos para indexação no sistema RAG.

**Content-Type**: `multipart/form-data`

**Parâmetros**:

| Campo           | Tipo        | Obrigatório | Descrição                                        |
| --------------- | ----------- | ----------- | ------------------------------------------------ |
| `file`          | File        | ✅          | Arquivo para upload                              |
| `namespace`     | String      | ❌          | Namespace para organização (padrão: "documents") |
| `chunk_size`    | Integer     | ❌          | Tamanho dos chunks (padrão: 1000)                |
| `chunk_overlap` | Integer     | ❌          | Sobreposição entre chunks (padrão: 200)          |
| `enable_cache`  | Boolean     | ❌          | Usar cache Redis (padrão: true)                  |
| `metadata`      | JSON String | ❌          | Metadados customizados                           |

**Formatos Suportados**:

- PDF (`.pdf`)
- Texto (`.txt`)
- Word (`.docx`)
- Markdown (`.md`)
- CSV (`.csv`)
- JSON (`.json`)

**Resposta de Sucesso** (200):

```json
{
  "success": true,
  "message": "Document processed successfully",
  "document_id": "doc_abc123",
  "namespace": "documents",
  "filename": "manual.pdf",
  "file_size_bytes": 2048576,
  "chunks_created": 15,
  "processing_time_ms": 3456,
  "metadata": {
    "pages": 42,
    "author": "Company Name",
    "created_date": "2024-01-15T10:30:00Z"
  }
}
```

**Exemplo de Uso**:

```bash
# Upload básico
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "namespace=legal-docs"

# Upload com parâmetros customizados
curl -X POST "http://localhost:8000/upload" \
  -F "file=@manual.pdf" \
  -F "namespace=user-manuals" \
  -F "chunk_size=1500" \
  -F "chunk_overlap=300" \
  -F 'metadata={"department":"engineering","version":"1.2"}'
```

### 🧠 RAG Query (Consulta Inteligente)

#### `POST /rag/query`

Executa consultas inteligentes usando o sistema RAG completo.

**Content-Type**: `application/json`

**Parâmetros**:

| Campo             | Tipo    | Obrigatório | Descrição                                       |
| ----------------- | ------- | ----------- | ----------------------------------------------- |
| `query`           | String  | ✅          | Pergunta do usuário                             |
| `namespace`       | String  | ❌          | Namespace específico para busca                 |
| `max_docs`        | Integer | ❌          | Máximo de documentos para retrieval (padrão: 5) |
| `model`           | String  | ❌          | Modelo LLM específico                           |
| `temperature`     | Float   | ❌          | Temperatura para geração (0.0-2.0)              |
| `max_tokens`      | Integer | ❌          | Máximo de tokens na resposta                    |
| `include_sources` | Boolean | ❌          | Incluir fontes na resposta (padrão: true)       |
| `filters`         | Object  | ❌          | Filtros de metadados para busca                 |

**Corpo da Requisição**:

```json
{
  "query": "Como configurar SSL no sistema?",
  "namespace": "technical-docs",
  "max_docs": 3,
  "model": "gpt-4o",
  "temperature": 0.1,
  "include_sources": true,
  "filters": {
    "document_type": "manual",
    "department": "engineering"
  }
}
```

**Resposta de Sucesso** (200):

```json
{
  "query": "Como configurar SSL no sistema?",
  "answer": "Para configurar SSL no sistema, siga estes passos:\n\n1. **Obter Certificados**: Use Let's Encrypt ou certificados próprios...",
  "sources": [
    {
      "document_id": "doc_xyz789",
      "filename": "ssl-setup-guide.pdf",
      "page": 12,
      "relevance_score": 0.95,
      "chunk_text": "Configuração SSL detalhada...",
      "metadata": {
        "author": "Security Team",
        "last_updated": "2024-01-10"
      }
    }
  ],
  "model_used": "gpt-4o",
  "processing_time_ms": 2341,
  "tokens_used": {
    "prompt": 1250,
    "completion": 890,
    "total": 2140
  },
  "cache_hit": false
}
```

**Exemplo de Uso**:

```bash
# Consulta básica
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Qual é o procedimento de backup?",
    "namespace": "operations"
  }'

# Consulta avançada com filtros
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Configuração de segurança",
    "namespace": "security-docs",
    "max_docs": 3,
    "model": "gpt-4o",
    "temperature": 0.0,
    "filters": {
      "classification": "internal",
      "date_range": {
        "start": "2024-01-01",
        "end": "2024-12-31"
      }
    }
  }'
```

### 💬 Chat Direto

#### `POST /chat`

Chat direto com modelos LLM sem sistema RAG.

**Content-Type**: `application/json`

**Parâmetros**:

| Campo             | Tipo    | Obrigatório | Descrição                               |
| ----------------- | ------- | ----------- | --------------------------------------- |
| `message`         | String  | ✅          | Mensagem do usuário                     |
| `system_prompt`   | String  | ❌          | Prompt de sistema customizado           |
| `model`           | String  | ❌          | Modelo específico (padrão: gpt-4o-mini) |
| `temperature`     | Float   | ❌          | Temperatura (padrão: 0.7)               |
| `max_tokens`      | Integer | ❌          | Máximo de tokens                        |
| `conversation_id` | String  | ❌          | ID para manter contexto                 |

**Corpo da Requisição**:

```json
{
  "message": "Explique como funciona o Docker",
  "system_prompt": "Você é um especialista em DevOps. Responda de forma técnica mas didática.",
  "model": "gpt-4o",
  "temperature": 0.3,
  "max_tokens": 1000
}
```

**Resposta de Sucesso** (200):

```json
{
  "response": "Docker é uma plataforma de containerização que permite...",
  "model_used": "gpt-4o",
  "tokens_used": {
    "prompt": 156,
    "completion": 445,
    "total": 601
  },
  "processing_time_ms": 1234,
  "conversation_id": "conv_abc123"
}
```

### 📊 Informações e Estatísticas

#### `GET /rag/info`

Informações sobre o pipeline RAG e configurações.

**Resposta**:

```json
{
  "pipeline_info": {
    "name": "rag_pipeline",
    "version": "2.0.0",
    "components": [
      "text_embedder",
      "retriever",
      "prompt_builder",
      "llm_generator"
    ]
  },
  "embeddings": {
    "model": "text-embedding-3-small",
    "dimensions": 1536,
    "total_embeddings": 15420
  },
  "vector_store": {
    "provider": "pinecone",
    "index": "haystack-docs",
    "total_vectors": 15420,
    "namespaces": ["documents", "legal-docs", "technical-specs"]
  },
  "models": {
    "available": ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"],
    "default": "gpt-4o-mini"
  }
}
```

#### `GET /cache/stats`

Estatísticas do cache Redis.

**Resposta**:

```json
{
  "redis_info": {
    "connected": true,
    "db_size": 1247,
    "memory_usage": "45.2MB",
    "hit_rate": 0.87,
    "total_hits": 12450,
    "total_misses": 1850
  },
  "cache_categories": {
    "embeddings": 890,
    "queries": 357,
    "documents": 0
  },
  "performance": {
    "avg_get_time_ms": 1.2,
    "avg_set_time_ms": 2.1
  }
}
```

#### `GET /api/models`

Lista de modelos disponíveis.

**Resposta**:

```json
{
  "models": {
    "llm": [
      {
        "id": "gpt-4o",
        "provider": "openai",
        "context_length": 128000,
        "supports_functions": true,
        "cost_per_1k_tokens": {
          "input": 0.005,
          "output": 0.015
        }
      },
      {
        "id": "claude-3-5-sonnet",
        "provider": "anthropic",
        "context_length": 200000,
        "supports_functions": false,
        "cost_per_1k_tokens": {
          "input": 0.003,
          "output": 0.015
        }
      }
    ],
    "embedding": [
      {
        "id": "text-embedding-3-small",
        "provider": "openai",
        "dimensions": 1536,
        "cost_per_1k_tokens": 0.00002
      }
    ]
  }
}
```

#### `GET /api/namespaces`

Lista de namespaces disponíveis.

**Resposta**:

```json
{
  "namespaces": [
    {
      "name": "documents",
      "document_count": 45,
      "total_chunks": 1250,
      "last_updated": "2024-01-15T09:30:00Z"
    },
    {
      "name": "legal-docs",
      "document_count": 12,
      "total_chunks": 450,
      "last_updated": "2024-01-14T16:20:00Z"
    }
  ],
  "total_namespaces": 2,
  "total_documents": 57,
  "total_chunks": 1700
}
```

#### `GET /api/documents`

Lista de documentos por namespace.

**Parâmetros de Query**:

- `namespace` (opcional): Filtrar por namespace
- `limit` (opcional): Número máximo de resultados (padrão: 50)
- `offset` (opcional): Offset para paginação (padrão: 0)

**Exemplo**: `GET /api/documents?namespace=legal-docs&limit=10`

**Resposta**:

```json
{
  "documents": [
    {
      "document_id": "doc_abc123",
      "filename": "contract_template.pdf",
      "namespace": "legal-docs",
      "upload_date": "2024-01-15T08:00:00Z",
      "file_size_bytes": 2048576,
      "chunks_count": 25,
      "metadata": {
        "pages": 15,
        "author": "Legal Team",
        "classification": "internal"
      }
    }
  ],
  "pagination": {
    "total": 12,
    "limit": 10,
    "offset": 0,
    "has_next": true
  }
}
```

## ⚙️ API Haystack Nativa (Porta 1416)

### 📋 Status dos Pipelines

#### `GET /status`

Status de todos os pipelines registrados.

**Resposta**:

```json
{
  "status": "ready",
  "pipelines": {
    "rag": {
      "status": "loaded",
      "components": 4,
      "last_run": "2024-01-15T10:25:00Z"
    },
    "chat": {
      "status": "loaded",
      "components": 2,
      "last_run": "2024-01-15T10:20:00Z"
    }
  },
  "health": {
    "memory_usage_mb": 512,
    "cpu_usage_percent": 15.3,
    "uptime_seconds": 3600
  }
}
```

#### `GET /`

Lista todos os pipelines disponíveis.

**Resposta**:

```json
{
  "pipelines": [
    {
      "name": "rag",
      "description": "Retrieval-Augmented Generation pipeline",
      "inputs": ["query", "namespace", "top_k"],
      "outputs": ["answer", "sources"]
    },
    {
      "name": "chat",
      "description": "Direct chat pipeline without RAG",
      "inputs": ["messages"],
      "outputs": ["response"]
    }
  ]
}
```

#### `GET /{pipeline_name}`

Detalhes de um pipeline específico.

**Exemplo**: `GET /rag`

**Resposta**:

```json
{
  "name": "rag",
  "description": "Retrieval-Augmented Generation pipeline for document Q&A",
  "components": [
    {
      "name": "text_embedder",
      "type": "OpenAITextEmbedder",
      "model": "text-embedding-3-small"
    },
    {
      "name": "retriever",
      "type": "PineconeEmbeddingRetriever",
      "top_k": 5
    },
    {
      "name": "prompt_builder",
      "type": "PromptBuilder",
      "template": "Context: {documents}\nQuestion: {query}\nAnswer:"
    },
    {
      "name": "llm",
      "type": "OpenAIGenerator",
      "model": "gpt-4o-mini"
    }
  ],
  "inputs": {
    "query": {
      "type": "string",
      "required": true,
      "description": "User question"
    },
    "namespace": {
      "type": "string",
      "required": false,
      "default": "documents",
      "description": "Document namespace"
    },
    "top_k": {
      "type": "integer",
      "required": false,
      "default": 5,
      "description": "Number of documents to retrieve"
    }
  },
  "outputs": {
    "answer": {
      "type": "string",
      "description": "Generated answer"
    },
    "sources": {
      "type": "list",
      "description": "Source documents used"
    }
  }
}
```

### 🧠 Executar Pipeline RAG

#### `POST /rag`

Executa o pipeline RAG nativo do Haystack.

**Content-Type**: `application/json`

**Corpo da Requisição**:

```json
{
  "query": "Como fazer backup do sistema?",
  "params": {
    "namespace": "operations",
    "top_k": 3
  }
}
```

**Resposta**:

```json
{
  "answer": "Para fazer backup do sistema, execute o script...",
  "sources": [
    {
      "content": "Script de backup automatizado...",
      "metadata": {
        "filename": "backup-guide.md",
        "score": 0.92
      }
    }
  ],
  "metadata": {
    "execution_time_ms": 1845,
    "pipeline": "rag",
    "component_times": {
      "text_embedder": 234,
      "retriever": 567,
      "prompt_builder": 12,
      "llm": 1032
    }
  }
}
```

### 💬 Executar Pipeline de Chat

#### `POST /chat`

Executa o pipeline de chat direto.

**Content-Type**: `application/json`

**Corpo da Requisição**:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Explique containerização"
    }
  ],
  "params": {
    "model": "gpt-4o",
    "temperature": 0.3
  }
}
```

**Resposta**:

```json
{
  "response": "Containerização é uma tecnologia...",
  "metadata": {
    "execution_time_ms": 1234,
    "pipeline": "chat",
    "tokens_used": 456
  }
}
```

## 📊 Códigos de Resposta

### Códigos de Sucesso

| Código | Status   | Descrição                                       |
| ------ | -------- | ----------------------------------------------- |
| 200    | OK       | Requisição processada com sucesso               |
| 201    | Created  | Recurso criado com sucesso                      |
| 202    | Accepted | Requisição aceita para processamento assíncrono |

### Códigos de Erro do Cliente

| Código | Status                 | Descrição                           | Solução                                          |
| ------ | ---------------------- | ----------------------------------- | ------------------------------------------------ |
| 400    | Bad Request            | Parâmetros inválidos ou malformados | Verificar formato JSON e parâmetros obrigatórios |
| 401    | Unauthorized           | API key inválida ou ausente         | Verificar configuração das API keys              |
| 403    | Forbidden              | Acesso negado ao recurso            | Verificar permissões e rate limits               |
| 404    | Not Found              | Recurso não encontrado              | Verificar URL e IDs de recursos                  |
| 413    | Payload Too Large      | Arquivo muito grande                | Reduzir tamanho do arquivo ou usar chunking      |
| 415    | Unsupported Media Type | Formato de arquivo não suportado    | Usar formatos suportados (PDF, TXT, DOCX, etc.)  |
| 422    | Unprocessable Entity   | Erro de validação nos dados         | Corrigir dados conforme esquema da API           |
| 429    | Too Many Requests      | Rate limit excedido                 | Aguardar ou reduzir frequência de requisições    |

### Códigos de Erro do Servidor

| Código | Status                | Descrição                            | Ação                                    |
| ------ | --------------------- | ------------------------------------ | --------------------------------------- |
| 500    | Internal Server Error | Erro interno do servidor             | Verificar logs, contatar suporte        |
| 502    | Bad Gateway           | Erro de proxy/gateway                | Verificar configuração de rede          |
| 503    | Service Unavailable   | Serviço temporariamente indisponível | Aguardar, verificar status dos serviços |
| 504    | Gateway Timeout       | Timeout na requisição                | Aumentar timeout ou otimizar consulta   |

### Estrutura de Erro Padrão

```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "Parameter 'query' is required",
    "details": {
      "parameter": "query",
      "received": null,
      "expected": "string"
    },
    "request_id": "req_abc123",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## 📋 Modelos de Dados

### Documento

```json
{
  "document_id": "string",
  "filename": "string",
  "namespace": "string",
  "upload_date": "datetime",
  "file_size_bytes": "integer",
  "chunks_count": "integer",
  "metadata": {
    "pages": "integer",
    "author": "string",
    "classification": "string",
    "custom_field": "any"
  }
}
```

### Consulta RAG

```json
{
  "query": "string",
  "namespace": "string (optional)",
  "max_docs": "integer (optional, default: 5)",
  "model": "string (optional)",
  "temperature": "float (optional, 0.0-2.0)",
  "max_tokens": "integer (optional)",
  "include_sources": "boolean (optional, default: true)",
  "filters": {
    "metadata_field": "value"
  }
}
```

### Resposta RAG

```json
{
  "query": "string",
  "answer": "string",
  "sources": [
    {
      "document_id": "string",
      "filename": "string",
      "page": "integer",
      "relevance_score": "float",
      "chunk_text": "string",
      "metadata": {}
    }
  ],
  "model_used": "string",
  "processing_time_ms": "integer",
  "tokens_used": {
    "prompt": "integer",
    "completion": "integer",
    "total": "integer"
  },
  "cache_hit": "boolean"
}
```

### Fonte de Documento

```json
{
  "document_id": "string",
  "filename": "string",
  "page": "integer (optional)",
  "relevance_score": "float (0.0-1.0)",
  "chunk_text": "string",
  "metadata": {
    "author": "string",
    "created_date": "datetime",
    "custom_fields": {}
  }
}
```

## 🔧 Exemplos de Integração

### Python

```python
import requests
import json

class HaystackRAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def upload_document(self, file_path, namespace="documents", **kwargs):
        """Upload a document to the RAG system"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'namespace': namespace, **kwargs}

            response = self.session.post(
                f"{self.base_url}/upload",
                files=files,
                data=data
            )
            return response.json()

    def query_rag(self, query, namespace=None, **kwargs):
        """Query the RAG system"""
        payload = {'query': query, **kwargs}
        if namespace:
            payload['namespace'] = namespace

        response = self.session.post(
            f"{self.base_url}/rag/query",
            json=payload
        )
        return response.json()

    def chat(self, message, **kwargs):
        """Direct chat without RAG"""
        payload = {'message': message, **kwargs}
        response = self.session.post(
            f"{self.base_url}/chat",
            json=payload
        )
        return response.json()

    def get_health(self):
        """Check system health"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()

# Exemplo de uso
client = HaystackRAGClient()

# Upload de documento
result = client.upload_document(
    "manual.pdf",
    namespace="user-manuals",
    chunk_size=1500
)
print(f"Upload: {result['message']}")

# Consulta RAG
answer = client.query_rag(
    "Como instalar o sistema?",
    namespace="user-manuals",
    max_docs=3
)
print(f"Resposta: {answer['answer']}")

# Chat direto
response = client.chat(
    "Explique a diferença entre Docker e containers"
)
print(f"Chat: {response['response']}")
```

### JavaScript/TypeScript

```javascript
class HaystackRAGClient {
  constructor(baseUrl = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  async uploadDocument(file, namespace = "documents", options = {}) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("namespace", namespace);

    Object.entries(options).forEach(([key, value]) => {
      formData.append(key, value);
    });

    const response = await fetch(`${this.baseUrl}/upload`, {
      method: "POST",
      body: formData,
    });

    return await response.json();
  }

  async queryRAG(query, options = {}) {
    const response = await fetch(`${this.baseUrl}/rag/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query, ...options }),
    });

    return await response.json();
  }

  async chat(message, options = {}) {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message, ...options }),
    });

    return await response.json();
  }

  async getHealth() {
    const response = await fetch(`${this.baseUrl}/health`);
    return await response.json();
  }
}

// Exemplo de uso
const client = new HaystackRAGClient();

// Upload de arquivo
const fileInput = document.getElementById("file-input");
const file = fileInput.files[0];

client
  .uploadDocument(file, "technical-docs")
  .then((result) => console.log("Upload:", result.message));

// Consulta RAG
client
  .queryRAG("Como configurar SSL?", {
    namespace: "technical-docs",
    max_docs: 3,
  })
  .then((result) => console.log("Resposta:", result.answer));
```

### cURL Scripts

```bash
#!/bin/bash
# upload.sh - Script para upload de documentos

API_BASE="http://localhost:8000"
FILE_PATH="$1"
NAMESPACE="${2:-documents}"

if [ -z "$FILE_PATH" ]; then
    echo "Uso: $0 <arquivo> [namespace]"
    exit 1
fi

echo "Fazendo upload de $FILE_PATH para namespace $NAMESPACE..."

curl -X POST "$API_BASE/upload" \
    -F "file=@$FILE_PATH" \
    -F "namespace=$NAMESPACE" \
    -F "chunk_size=1000" \
    -F "chunk_overlap=200" \
    | jq '.'
```

```bash
#!/bin/bash
# query.sh - Script para consultas RAG

API_BASE="http://localhost:8000"
QUERY="$1"
NAMESPACE="$2"

if [ -z "$QUERY" ]; then
    echo "Uso: $0 <pergunta> [namespace]"
    exit 1
fi

PAYLOAD="{\"query\": \"$QUERY\""
if [ -n "$NAMESPACE" ]; then
    PAYLOAD="$PAYLOAD, \"namespace\": \"$NAMESPACE\""
fi
PAYLOAD="$PAYLOAD}"

echo "Consultando: $QUERY"

curl -X POST "$API_BASE/rag/query" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    | jq '.answer'
```

### Webhooks (Opcional)

Para notificações assíncronas:

```json
// POST /api/webhooks (configuração)
{
  "url": "https://your-app.com/webhook",
  "events": ["document.uploaded", "document.processed"],
  "secret": "webhook-secret-key"
}

// Payload de webhook recebido
{
  "event": "document.processed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "document_id": "doc_abc123",
    "namespace": "documents",
    "status": "completed",
    "chunks_created": 15
  },
  "signature": "sha256=signature-hash"
}
```

## ⚡ Rate Limiting

### Limites Padrão

| Endpoint     | Limite       | Janela   | Burst |
| ------------ | ------------ | -------- | ----- |
| `/upload`    | 10 req/min   | 1 minuto | 5     |
| `/rag/query` | 60 req/min   | 1 minuto | 20    |
| `/chat`      | 100 req/min  | 1 minuto | 30    |
| `/health`    | 1000 req/min | 1 minuto | 100   |

### Headers de Rate Limit

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642248600
X-RateLimit-Retry-After: 60
```

### Resposta de Rate Limit Excedido

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 60 seconds",
    "details": {
      "limit": 60,
      "window": "1 minute",
      "retry_after": 60
    }
  }
}
```

## 🚨 Troubleshooting

### Problemas Comuns

#### 1. Erro 401 - Unauthorized

**Problema**: API keys não configuradas ou inválidas.

**Solução**:

```bash
# Verificar configuração
docker-compose exec hayhooks env | grep API_KEY

# Testar conectividade
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
    https://api.openai.com/v1/models
```

#### 2. Erro 413 - Payload Too Large

**Problema**: Arquivo muito grande para upload.

**Solução**:

```bash
# Verificar limite atual
curl -I http://localhost:8000/upload

# Configurar limite maior no nginx
client_max_body_size 100M;
```

#### 3. Erro 503 - Service Unavailable

**Problema**: Serviços dependentes indisponíveis.

**Solução**:

```bash
# Verificar status dos serviços
docker-compose ps

# Verificar logs
docker-compose logs hayhooks redis

# Restart se necessário
docker-compose restart
```

#### 4. Timeout em Consultas

**Problema**: Consultas RAG muito lentas.

**Solução**:

```bash
# Verificar cache
curl http://localhost:8000/cache/stats

# Otimizar parâmetros
{
  "query": "pergunta",
  "max_docs": 3,  // Reduzir número de documentos
  "namespace": "specific"  // Usar namespace específico
}
```

### Debug Avançado

```bash
# Logs detalhados
docker-compose logs -f hayhooks | grep ERROR

# Monitoramento de performance
curl http://localhost:8000/api/metrics

# Teste de conectividade completo
curl -X POST "http://localhost:8000/debug/connectivity" \
    -H "Content-Type: application/json" \
    -d '{"test_all": true}'
```

### Contato e Suporte

Para problemas não resolvidos:

1. **Logs de Debug**: Sempre colete logs antes de reportar
2. **Versão**: Inclua versão do sistema e componentes
3. **Configuração**: Remova informações sensíveis (API keys)
4. **Reprodução**: Descreva passos para reproduzir o problema

```bash
# Script de coleta de informações
./scripts/collect-debug-info.sh > debug-info.txt
```

---

## 📚 Referências

- **OpenAPI Specification**: [https://spec.openapis.org/oas/v3.0.3](https://spec.openapis.org/oas/v3.0.3)
- **Haystack Documentation**: [https://docs.haystack.deepset.ai/](https://docs.haystack.deepset.ai/)
- **FastAPI Documentation**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **OpenAI API Reference**: [https://platform.openai.com/docs/api-reference](https://platform.openai.com/docs/api-reference)
- **Pinecone API Docs**: [https://docs.pinecone.io/](https://docs.pinecone.io/)

---

**Versão da API**: 2.0.0  
**Última Atualização**: Janeiro 2024  
**Compatibilidade**: Haystack RAG System v2.0+

Para documentação interativa completa, acesse: `http://localhost:8000/docs`
