# Multi-RAG Engine

Sistema avançado de Retrieval-Augmented Generation (RAG) com suporte multi-modelo, visualização de pipelines e interface web integrada, construído com Haystack 2.0.

## 🚀 Características

- **Multi-Modelo**: Suporte para OpenAI, Anthropic, e outros provedores
- **Visualização de Pipelines**: Interface web com diagramas Mermaid
- **RAG Avançado**: Sistema completo de indexação e consulta de documentos
- **Chat Inteligente**: Interface conversacional com histórico
- **API RESTful**: Endpoints FastAPI para integração
- **Docker Ready**: Configuração completa com Docker Compose
- **Monitoramento**: Health checks e logging integrados

## 🏗️ Arquitetura

### Componentes Principais

- **Hayhooks**: API principal com pipelines Haystack
- **Pipeline Visualizer**: Sistema de visualização e gerenciamento
- **Qdrant**: Vector database para armazenamento de embeddings
- **Redis**: Cache e sessões
- **Mermaid Visualizer**: Serviço para geração de diagramas

### Pipelines Implementados

1. **Indexing Pipeline**: Processa e indexa documentos
2. **RAG Pipeline**: Busca e geração de respostas
3. **Chat Pipeline**: Conversação com contexto e histórico

## 🛠️ Tecnologias

- **Framework**: Haystack 2.0
- **API**: FastAPI + Hayhooks
- **Vector DB**: Qdrant
- **Cache**: Redis
- **Frontend**: HTML5 + JavaScript (Vanilla)
- **Containerização**: Docker + Docker Compose
- **Visualização**: Mermaid.js

## 📋 Pré-requisitos

- Docker e Docker Compose
- Python 3.11+ (para desenvolvimento local)
- Chaves API dos provedores de LLM (OpenAI, Anthropic, etc.)

## 🚀 Instalação e Uso

### 1. Clone o Repositório

```bash
git clone https://github.com/brunoaquino/multi-rag-engine.git
cd multi-rag-engine
```

### 2. Configuração de Variáveis

```bash
cp .env.example .env
# Edite o .env com suas chaves API
```

### 3. Executar com Docker

```bash
docker-compose up -d
```

### 4. Acessar Interfaces

- **API Principal**: http://localhost:1416
- **Visualizador**: http://localhost:1416/visualize/interface
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Redis Insight**: http://localhost:8001

## 📚 Documentação da API

### Endpoints Principais

#### Pipeline Management

- `GET /status` - Status dos pipelines
- `POST /indexing` - Indexar documentos
- `POST /query` - Consultar documentos
- `POST /chat` - Chat conversacional

#### Visualização

- `GET /visualize/pipelines` - Listar pipelines
- `GET /visualize/pipeline/{name}/info` - Informações do pipeline
- `GET /visualize/pipeline/{name}/visualize` - Gerar visualização
- `GET /visualize/interface` - Interface web

### Exemplos de Uso

#### Indexar Documento

```bash
curl -X POST "http://localhost:1416/indexing" \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["./data/document.pdf"],
    "meta": {"source": "upload"}
  }'
```

#### Consultar Documentos

```bash
curl -X POST "http://localhost:1416/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Qual é o conteúdo principal do documento?",
    "params": {"Retriever": {"top_k": 5}}
  }'
```

#### Chat

```bash
curl -X POST "http://localhost:1416/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Olá, como posso consultar documentos?",
    "session_id": "user123"
  }'
```

## 🔧 Desenvolvimento

### Estrutura do Projeto

```
multi-rag-engine/
├── hayhooks/                  # Código principal da API
│   ├── server/               # Configuração do servidor
│   ├── pipeline_visualizer.py # Sistema de visualização
│   ├── visualization_endpoints.py # Endpoints de visualização
│   └── model_manager.py      # Gerenciamento multi-modelo
├── pipelines/                # Definições de pipelines YAML
├── data/                     # Dados e documentos
├── static/                   # Assets estáticos
├── docker-compose.yml        # Orquestração de containers
└── README.md                # Esta documentação
```

### Executar Localmente

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar servidor de desenvolvimento
python -m hayhooks.server.main

# Executar testes
pytest tests/
```

## 🐳 Docker

### Configuração Completa

O projeto inclui configuração Docker Compose com:

- **Hayhooks**: API principal (porta 1416)
- **Qdrant**: Vector database (porta 6333)
- **Redis**: Cache (porta 6379)
- **Mermaid Visualizer**: Geração de diagramas (porta 9000)

### Variáveis de Ambiente

```env
# LLM APIs
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379

# Logging
LOG_LEVEL=INFO
```

## 📊 Recursos Avançados

### Gerenciamento Multi-Modelo

- **Fallback Chains**: Estratégias de fallback automático
- **Cost Optimization**: Otimização de custos baseada em uso
- **Rate Limiting**: Controle de taxa por provedor
- **Health Monitoring**: Monitoramento de saúde dos provedores

### Visualização de Pipelines

- **Diagramas Mermaid**: Visualização automática de fluxos
- **Interface Web**: Gestão visual de pipelines
- **Export/Import**: Salvar e carregar configurações
- **Debugging**: Ferramentas de debug visual

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- **Haystack Team** - Framework principal
- **deepset** - Ferramentas e documentação
- **Qdrant** - Vector database
- **FastAPI** - Framework web

## 📞 Suporte

- **Issues**: [GitHub Issues](https://github.com/brunoaquino/multi-rag-engine/issues)
- **Documentação**: [Wiki do Projeto](https://github.com/brunoaquino/multi-rag-engine/wiki)

---

Desenvolvido com ❤️ usando Haystack 2.0
