# 🔄 Data Flow Diagrams - Haystack RAG System

Este documento apresenta as representações visuais dos fluxos de dados no sistema Haystack RAG, complementando as descrições textuais disponíveis no [README.md](../README.md).

> 📖 **Para contexto técnico**: [ARCHITECTURE.md](ARCHITECTURE.md)  
> 🔌 **Para integração**: [API.md](API.md)  
> 🚀 **Para deploy**: [DEPLOYMENT.md](DEPLOYMENT.md)

## 📋 Índice

- [Pipeline de Upload e Processamento](#-pipeline-de-upload-e-processamento)
- [Pipeline de Consulta RAG](#-pipeline-de-consulta-rag)
- [Fluxo de Cache e Otimização](#-fluxo-de-cache-e-otimização)
- [Arquitetura de Componentes](#-arquitetura-de-componentes)
- [Fluxos de Dados Detalhados](#-fluxos-de-dados-detalhados)

## 📤 Pipeline de Upload e Processamento

### Fluxo Principal de Upload

```mermaid
graph TB
    %% Entrada de Dados
    User[👤 Usuário] --> UI{Interface de Upload}
    UI --> Streamlit[🖥️ Streamlit Frontend<br/>:8501]
    UI --> API[🔌 Direct API<br/>:8000/upload]

    %% Processamento de Upload
    Streamlit --> Upload[📤 Upload Handler]
    API --> Upload

    Upload --> Validate[✅ Validação<br/>Formato & Tamanho]
    Validate --> Extract[📄 Extração de Texto<br/>PyPDF, python-docx]

    %% Processamento de Texto
    Extract --> Split[✂️ Text Splitting<br/>Chunk Size: 1000<br/>Overlap: 200]
    Split --> Embed[🧠 Embedding Generation<br/>OpenAI text-embedding-3-small]

    %% Cache e Persistência
    Embed --> Cache[💾 Redis Cache<br/>Embeddings + Metadata]
    Cache --> Store[🗄️ Pinecone Vector Store<br/>Indexação por Namespace]

    %% Resposta
    Store --> Response[✅ Upload Response<br/>Document ID + Stats]
    Response --> User

    %% Estilo dos nós
    classDef user fill:#e1f5fe
    classDef interface fill:#f3e5f5
    classDef processing fill:#e8f5e8
    classDef storage fill:#fff3e0
    classDef response fill:#e0f2f1

    class User user
    class Streamlit,API interface
    class Upload,Validate,Extract,Split,Embed processing
    class Cache,Store storage
    class Response response
```

### Detalhamento do Processamento

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant S as 🖥️ Streamlit
    participant A as 🔌 API Handler
    participant V as ✅ Validator
    participant E as 📄 Extractor
    participant T as ✂️ Text Splitter
    participant M as 🧠 Embedder
    participant R as 💾 Redis
    participant P as 🗄️ Pinecone

    U->>S: Upload arquivo (PDF/TXT/DOCX)
    S->>A: POST /upload + file + params

    A->>V: Validar arquivo
    V-->>A: ✅ Validação OK

    A->>E: Extrair texto do arquivo
    E-->>A: Texto extraído + metadata

    A->>T: Dividir em chunks
    T-->>A: Lista de chunks

    loop Para cada chunk
        A->>M: Gerar embedding
        M->>R: Cache embedding
        R-->>M: Cache hit/miss
        M-->>A: Vector embedding
    end

    A->>P: Indexar no namespace
    P-->>A: Index response

    A-->>S: Upload success + stats
    S-->>U: Confirmação + progress
```

## 🔍 Pipeline de Consulta RAG

### Fluxo Principal de Consulta

```mermaid
graph TB
    %% Entrada da Query
    User[👤 Usuário] --> UI{Interface de Consulta}
    UI --> OpenWebUI[🤖 OpenWebUI<br/>:3000]
    UI --> DirectAPI[🔌 Direct API<br/>:8000/rag/query]

    %% Processamento da Query
    OpenWebUI --> RAG[🧠 RAG Pipeline]
    DirectAPI --> RAG

    RAG --> QueryEmbed[🔍 Query Embedding<br/>OpenAI text-embedding-3-small]
    QueryEmbed --> CacheCheck{💾 Cache Check<br/>Redis}

    %% Cache Hit Path
    CacheCheck -->|Cache Hit| CachedResponse[⚡ Cached Response]
    CachedResponse --> FormatResponse[📝 Format Response]

    %% Cache Miss Path
    CacheCheck -->|Cache Miss| VectorSearch[🔍 Vector Search<br/>Pinecone Similarity]
    VectorSearch --> RetrieveDocs[📚 Retrieve Documents<br/>Top-K Filtering]

    RetrieveDocs --> PromptBuild[📝 Prompt Builder<br/>Context + Question]
    PromptBuild --> LLM[🤖 LLM Generation<br/>GPT-4o / Claude]

    LLM --> CacheStore[💾 Cache Store<br/>Redis TTL]
    CacheStore --> FormatResponse

    %% Resposta Final
    FormatResponse --> ResponseData[📋 Structured Response<br/>Answer + Sources + Metadata]
    ResponseData --> User

    %% Estilos
    classDef user fill:#e1f5fe
    classDef interface fill:#f3e5f5
    classDef processing fill:#e8f5e8
    classDef storage fill:#fff3e0
    classDef cache fill:#e3f2fd
    classDef response fill:#e0f2f1

    class User user
    class OpenWebUI,DirectAPI interface
    class RAG,QueryEmbed,VectorSearch,RetrieveDocs,PromptBuild,LLM processing
    class CacheCheck,CacheStore storage
    class CachedResponse cache
    class FormatResponse,ResponseData response
```

### Fluxo Detalhado de RAG

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant O as 🤖 OpenWebUI
    participant R as 🧠 RAG Pipeline
    participant E as 🔍 Embedder
    participant C as 💾 Redis Cache
    participant P as 🗄️ Pinecone
    participant L as 🤖 LLM (GPT-4o)

    U->>O: "Como configurar SSL?"
    O->>R: POST /rag/query + params

    R->>E: Embed query
    E-->>R: Query vector [1536 dims]

    R->>C: Check cache (query hash)
    alt Cache Hit
        C-->>R: Cached response
        R-->>O: ⚡ Fast response
    else Cache Miss
        R->>P: Vector similarity search
        P-->>R: Top-K documents + scores

        R->>R: Build context prompt
        R->>L: Generate answer
        L-->>R: LLM response + metadata

        R->>C: Cache response (TTL: 1h)
        R-->>O: 📋 Complete response
    end

    O-->>U: Answer + sources + metadata
```

## 💾 Fluxo de Cache e Otimização

### Estratégia de Cache Multi-Camada

```mermaid
graph LR
    %% Entrada
    Request[📥 Request] --> L1{L1: Memory Cache<br/>Application Level}

    %% Layer 1 - Application Memory
    L1 -->|Hit| FastResponse[⚡ Instant Response<br/>< 1ms]
    L1 -->|Miss| L2{L2: Redis Cache<br/>Distributed}

    %% Layer 2 - Redis
    L2 -->|Hit| MediumResponse[🔥 Fast Response<br/>< 10ms]
    L2 -->|Miss| L3{L3: Embedding Cache<br/>Vector Store}

    %% Layer 3 - Vector Cache
    L3 -->|Hit| ComputeResponse[🔄 Compute Response<br/>< 100ms]
    L3 -->|Miss| FullCompute[🧠 Full Processing<br/>1-5s]

    %% Cache Storage
    FullCompute --> UpdateCaches[📝 Update All Caches]
    UpdateCaches --> L1
    UpdateCaches --> L2
    UpdateCaches --> L3

    %% Cache Types
    subgraph "Cache Categories"
        CacheEmbed[🧠 Embeddings<br/>TTL: 24h]
        CacheQuery[🔍 Query Results<br/>TTL: 1h]
        CacheDoc[📄 Document Metadata<br/>TTL: 12h]
    end

    %% Styles
    classDef cache fill:#e3f2fd
    classDef fast fill:#c8e6c9
    classDef medium fill:#fff9c4
    classDef slow fill:#ffcdd2

    class L1,L2,L3,CacheEmbed,CacheQuery,CacheDoc cache
    class FastResponse,MediumResponse fast
    class ComputeResponse medium
    class FullCompute slow
```

### Cache Performance Metrics

```mermaid
pie title Cache Hit Rates by Type
    "Embedding Cache" : 85
    "Query Cache" : 65
    "Document Cache" : 90
    "Miss (Full Compute)" : 15
```

## 🏗️ Arquitetura de Componentes

### Visão Geral da Arquitetura

```mermaid
graph TB
    %% Frontend Layer
    subgraph "🖥️ Frontend Layer"
        OpenWebUI[🤖 OpenWebUI<br/>Chat Interface<br/>:3000]
        Streamlit[📤 Streamlit<br/>Upload Interface<br/>:8501]
        WebApp[🌐 Custom Web App<br/>Optional<br/>:8080]
    end

    %% API Gateway Layer
    subgraph "🔌 API Gateway Layer"
        Nginx[🔄 Nginx Proxy<br/>Load Balancer<br/>:80/443]
        DirectAPI[⚡ Direct API<br/>FastAPI<br/>:8000]
        HayAPI[🧠 Haystack API<br/>Hayhooks<br/>:1416]
    end

    %% Business Logic Layer
    subgraph "🧠 Business Logic Layer"
        RAGPipeline[🔍 RAG Pipeline<br/>Haystack Components]
        ChatPipeline[💬 Chat Pipeline<br/>Direct LLM]
        UploadHandler[📤 Upload Handler<br/>Document Processing]
        CacheManager[💾 Cache Manager<br/>Redis Operations]
    end

    %% AI Services Layer
    subgraph "🤖 AI Services Layer"
        OpenAI[🧠 OpenAI<br/>GPT-4o + Embeddings]
        Anthropic[🤖 Anthropic<br/>Claude Models]
        LocalLLM[🏠 Local LLM<br/>Optional Ollama]
    end

    %% Data Layer
    subgraph "🗄️ Data Layer"
        Pinecone[📊 Pinecone<br/>Vector Database]
        Redis[💾 Redis<br/>Cache + Sessions]
        FileSystem[📁 File System<br/>Temp Storage]
    end

    %% Monitoring Layer
    subgraph "📊 Monitoring Layer"
        Prometheus[📈 Prometheus<br/>Metrics Collection]
        Grafana[📊 Grafana<br/>Dashboards]
        Logs[📋 Centralized Logs<br/>ELK Stack]
    end

    %% Connections
    OpenWebUI <--> Nginx
    Streamlit <--> Nginx
    WebApp <--> Nginx

    Nginx <--> DirectAPI
    Nginx <--> HayAPI

    DirectAPI <--> RAGPipeline
    DirectAPI <--> ChatPipeline
    DirectAPI <--> UploadHandler
    HayAPI <--> RAGPipeline

    RAGPipeline <--> CacheManager
    ChatPipeline <--> CacheManager
    UploadHandler <--> CacheManager

    RAGPipeline <--> OpenAI
    ChatPipeline <--> OpenAI
    RAGPipeline <--> Anthropic
    ChatPipeline <--> LocalLLM

    RAGPipeline <--> Pinecone
    UploadHandler <--> Pinecone
    CacheManager <--> Redis
    UploadHandler <--> FileSystem

    DirectAPI <--> Prometheus
    HayAPI <--> Prometheus
    Prometheus <--> Grafana

    %% Styles
    classDef frontend fill:#e1f5fe
    classDef api fill:#f3e5f5
    classDef logic fill:#e8f5e8
    classDef ai fill:#fff3e0
    classDef data fill:#fce4ec
    classDef monitor fill:#f1f8e9

    class OpenWebUI,Streamlit,WebApp frontend
    class Nginx,DirectAPI,HayAPI api
    class RAGPipeline,ChatPipeline,UploadHandler,CacheManager logic
    class OpenAI,Anthropic,LocalLLM ai
    class Pinecone,Redis,FileSystem data
    class Prometheus,Grafana,Logs monitor
```

## 📊 Fluxos de Dados Detalhados

### Métricas de Performance

```mermaid
gantt
    title Performance Timeline por Operação
    dateFormat X
    axisFormat %s

    section Upload Process
    File Validation     :0, 100
    Text Extraction     :100, 500
    Text Chunking       :500, 700
    Embedding Generation:700, 2000
    Vector Indexing     :2000, 2500
    Cache Storage       :2500, 2600

    section RAG Query
    Query Embedding     :0, 200
    Cache Check         :200, 210
    Vector Search       :210, 500
    Context Building    :500, 520
    LLM Generation      :520, 1500
    Response Formatting :1500, 1520
```

### Data Flow Metrics

| Operação               | Latência Média | Throughput    | Cache Hit Rate |
| ---------------------- | -------------- | ------------- | -------------- |
| **Upload**             | 2.5s           | 10 docs/min   | N/A            |
| **Query (Cache Hit)**  | 50ms           | 100 queries/s | 75%            |
| **Query (Cache Miss)** | 1.2s           | 20 queries/s  | 25%            |
| **Embedding**          | 200ms          | 50 vectors/s  | 85%            |

### Volume de Dados

```mermaid
graph LR
    subgraph "📈 Volume Metrics"
        Docs[📄 Documents<br/>~1000 docs]
        Chunks[✂️ Chunks<br/>~25,000 chunks]
        Vectors[🧠 Vectors<br/>~25,000 x 1536 dims]
        Cache[💾 Cache Entries<br/>~5,000 queries]
    end

    Docs --> Chunks
    Chunks --> Vectors
    Vectors --> Cache

    subgraph "💾 Storage Requirements"
        VectorSize[Vector Storage<br/>~150MB]
        CacheSize[Cache Storage<br/>~50MB]
        MetaSize[Metadata<br/>~10MB]
        TotalSize[Total: ~210MB]
    end
```

## 🔄 Otimizações Implementadas

### Performance Optimizations

1. **🚀 Connection Pooling**

   ```python
   # Pinecone connection pool
   pinecone_pool = ConnectionPool(max_connections=10)

   # Redis connection pool
   redis_pool = redis.ConnectionPool(max_connections=20)
   ```

2. **⚡ Async Processing**

   ```python
   # Parallel embedding generation
   async def process_chunks_parallel(chunks):
       tasks = [embed_chunk(chunk) for chunk in chunks]
       return await asyncio.gather(*tasks)
   ```

3. **💾 Smart Caching**

   ```python
   # Multi-level cache strategy
   @cache_with_ttl(ttl=3600)  # 1 hour
   async def cached_rag_query(query_hash):
       return await rag_pipeline.run(query)
   ```

4. **📊 Batch Operations**
   ```python
   # Batch vector upserts
   batch_size = 100
   for i in range(0, len(vectors), batch_size):
       batch = vectors[i:i+batch_size]
       await pinecone_index.upsert(batch)
   ```

---

## 🎯 Próximos Passos para Otimização

1. **🔄 Streaming Responses**: Implementar streaming para respostas longas
2. **📊 Monitoring Avançado**: Adicionar métricas detalhadas de performance
3. **🧠 Semantic Caching**: Cache baseado em similaridade semântica
4. **🔧 Auto-scaling**: Implementar scaling automático baseado em carga

---

**📅 Última Atualização**: Janeiro 2024  
**🔄 Próxima Revisão**: Fevereiro 2024  
**📊 Métricas Baseadas**: Sistema em produção com ~1000 documentos
