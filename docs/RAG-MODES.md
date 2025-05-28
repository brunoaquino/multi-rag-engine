# 🧠 RAG Modes Guide - Intelligent vs Document-Only Responses

Este guia explica os diferentes modos de operação do sistema RAG (Retrieval-Augmented Generation) e como escolher o melhor para cada situação.

> 📖 **Para setup básico**: [README.md](../README.md)  
> 🚀 **Para deploy**: [DEPLOYMENT.md](DEPLOYMENT.md)  
> 🔌 **Para APIs**: [API.md](API.md)

## 📋 Índice

- [Visão Geral dos Modos](#-visão-geral-dos-modos)
- [Modo Strict (Apenas Documentos)](#-modo-strict-apenas-documentos)
- [Modo Hybrid (Recomendado)](#-modo-hybrid-recomendado)
- [Comparação Prática](#-comparação-prática)
- [Como Configurar](#-como-configurar)
- [Exemplos de Uso](#-exemplos-de-uso)
- [Casos de Uso Recomendados](#-casos-de-uso-recomendados)

## 🎯 Visão Geral dos Modos

O sistema RAG oferece diferentes modos de operação para atender diferentes necessidades:

| Modo         | Descrição                                   | Quando Usar                                             |
| ------------ | ------------------------------------------- | ------------------------------------------------------- |
| **Strict**   | Apenas informações dos documentos indexados | Compliance, documentação específica, respostas precisas |
| **Hybrid**   | Combina documentos + conhecimento global    | Uso geral, educação, respostas completas                |
| **Enhanced** | Modo avançado (futuro)                      | Atualmente mapeia para hybrid                           |

## 🔒 Modo Strict (Apenas Documentos)

### Características

- ✅ **Precisão máxima**: Apenas informações dos documentos
- ✅ **Rastreabilidade**: Todas as respostas são baseadas em fontes conhecidas
- ✅ **Compliance**: Ideal para ambientes regulamentados
- ❌ **Limitado**: Pode não responder perguntas fora do escopo dos documentos

### Template de Prompt

```
You are a document-based AI assistant. You must answer questions ONLY using information from the provided context.

STRICT INSTRUCTIONS:
- Answer ONLY using information explicitly stated in the provided context
- If the context doesn't contain the information needed, respond with "I cannot answer this question based on the provided documents."
- Do NOT use any external knowledge or make assumptions beyond what's in the context
- Quote or reference specific parts of the context when possible
- Be precise and factual
```

### Exemplo de Resposta

**Pergunta**: "Como implementar autenticação JWT?"

**Resposta Strict**: "I cannot answer this question based on the provided documents. The indexed documents do not contain specific information about JWT authentication implementation."

## 🔄 Modo Hybrid (Recomendado)

### Características

- ✅ **Completo**: Combina documentos + conhecimento global
- ✅ **Inteligente**: Enriquece respostas quando necessário
- ✅ **Transparente**: Distingue claramente entre fontes
- ✅ **Útil**: Fornece respostas mais abrangentes e práticas

### Template de Prompt

```
You are an intelligent AI assistant with access to both specific document context and general knowledge.

INSTRUCTIONS:
1. **Primary Focus**: Use the provided document context as your main information source
2. **Context Integration**: When the documents contain relevant information, cite and reference them specifically
3. **Knowledge Enhancement**: If the document context is incomplete or would benefit from additional context, supplement with your general knowledge to provide a more comprehensive answer
4. **Clear Attribution**:
   - When using document information, indicate this clearly (e.g., "According to the provided documents...")
   - When adding general knowledge, make this distinction (e.g., "Additionally, from general knowledge...")
5. **Balanced Response**: Aim for responses that are both grounded in the provided context and helpfully complete
```

### Exemplo de Resposta

**Pergunta**: "Como implementar autenticação JWT?"

**Resposta Hybrid**:
"According to the provided documents, there are references to authentication systems, but specific JWT implementation details are not covered.

Additionally, from general knowledge, JWT (JSON Web Token) authentication can be implemented as follows:

1. **Token Generation**: Create a JWT containing user claims
2. **Token Validation**: Verify the token signature and expiration
3. **Middleware Integration**: Add JWT validation to your API routes

Here's a basic implementation example:

```python
import jwt
from datetime import datetime, timedelta

def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, 'secret_key', algorithm='HS256')
```

For your specific system, you would need to integrate this with the authentication patterns mentioned in your documentation."

## 📊 Comparação Prática

### Teste com Pergunta Técnica

**Pergunta**: "Quais são as melhores práticas para otimização de performance em Python?"

| Aspecto        | Modo Strict          | Modo Hybrid                                |
| -------------- | -------------------- | ------------------------------------------ |
| **Resposta**   | "I cannot answer..." | Resposta completa com práticas específicas |
| **Utilidade**  | ⭐⭐                 | ⭐⭐⭐⭐⭐                                 |
| **Precisão**   | ⭐⭐⭐⭐⭐           | ⭐⭐⭐⭐                                   |
| **Completude** | ⭐⭐                 | ⭐⭐⭐⭐⭐                                 |

### Teste com Pergunta Específica dos Documentos

**Pergunta**: "Como configurar o sistema de cache Redis neste projeto?"

| Aspecto        | Modo Strict                        | Modo Hybrid                                 |
| -------------- | ---------------------------------- | ------------------------------------------- |
| **Resposta**   | Informações específicas do projeto | Informações do projeto + contexto adicional |
| **Utilidade**  | ⭐⭐⭐⭐                           | ⭐⭐⭐⭐⭐                                  |
| **Precisão**   | ⭐⭐⭐⭐⭐                         | ⭐⭐⭐⭐⭐                                  |
| **Completude** | ⭐⭐⭐                             | ⭐⭐⭐⭐⭐                                  |

## ⚙️ Como Configurar

### 1. Configuração Padrão (Hybrid)

O sistema já vem configurado em modo hybrid por padrão:

```python
from rag_pipeline import create_rag_pipeline

# Cria pipeline em modo hybrid (padrão)
rag = create_rag_pipeline(
    pinecone_index="haystack-rag",
    rag_mode="hybrid"  # Padrão
)
```

### 2. Configuração via API

```bash
# Modo hybrid (padrão)
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Como funciona JWT?",
    "rag_mode": "hybrid"
  }'

# Modo strict
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "O que está nos nossos documentos sobre JWT?",
    "rag_mode": "strict"
  }'
```

### 3. Configuração via Python

```python
import requests

# Query em modo hybrid
response = requests.post("http://localhost:8000/rag/query", json={
    "question": "Como implementar cache Redis?",
    "rag_mode": "hybrid"
})

# Query em modo strict
response = requests.post("http://localhost:8000/rag/query", json={
    "question": "Configuração específica do Redis no projeto",
    "rag_mode": "strict"
})
```

## 💡 Exemplos de Uso

### Exemplo 1: Pergunta Geral (Hybrid Recomendado)

```python
# Pergunta que se beneficia de conhecimento geral
query = {
    "question": "Quais são as melhores práticas de segurança para APIs REST?",
    "rag_mode": "hybrid"
}

# Resposta combinará:
# 1. Informações específicas dos documentos do projeto
# 2. Melhores práticas gerais de segurança
# 3. Exemplos práticos de implementação
```

### Exemplo 2: Pergunta Específica (Strict Recomendado)

```python
# Pergunta sobre configuração específica do projeto
query = {
    "question": "Qual é a configuração exata do Redis neste projeto?",
    "rag_mode": "strict"
}

# Resposta será baseada apenas nos documentos indexados
# Garantindo precisão total das configurações
```

### Exemplo 3: Pergunta Educacional (Hybrid Recomendado)

```python
# Pergunta educacional que precisa de contexto amplo
query = {
    "question": "Como funciona o algoritmo de embedding usado no sistema?",
    "rag_mode": "hybrid"
}

# Resposta combinará:
# 1. Detalhes específicos do modelo usado no projeto
# 2. Explicação geral de como embeddings funcionam
# 3. Contexto sobre por que essa escolha foi feita
```

## 🎯 Casos de Uso Recomendados

### Use Modo **Strict** quando:

- ✅ **Compliance e Auditoria**: Respostas devem ser rastreáveis
- ✅ **Documentação Técnica**: Informações específicas do projeto
- ✅ **Configurações Críticas**: Evitar interpretações incorretas
- ✅ **Ambientes Regulamentados**: Setor financeiro, saúde, legal

### Use Modo **Hybrid** quando:

- ✅ **Suporte Geral**: Ajuda aos usuários com contexto amplo
- ✅ **Educação**: Explicações completas e didáticas
- ✅ **Desenvolvimento**: Orientações práticas de implementação
- ✅ **Troubleshooting**: Soluções baseadas em experiência geral

## 🧪 Testando os Modos

### Script de Teste Automático

```bash
# Execute o script de teste para comparar os modos
python scripts/test_rag_modes.py
```

### Teste Manual via API

```bash
# Teste modo hybrid
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Como otimizar performance de aplicações Python?",
    "rag_mode": "hybrid"
  }'

# Teste modo strict
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Como otimizar performance de aplicações Python?",
    "rag_mode": "strict"
  }'
```

### Configuração Personalizada

```bash
# Configure seu pipeline para modo hybrid otimizado
python scripts/configure_hybrid_rag.py
```

## 📈 Métricas e Monitoramento

### Indicadores de Qualidade

- **Strict Mode**:
  - Taxa de "não posso responder": Indica cobertura dos documentos
  - Precisão das citações: Verificar se as fontes estão corretas
- **Hybrid Mode**:
  - Balanceamento fonte/conhecimento: Proporção entre informações dos docs vs. conhecimento geral
  - Satisfação do usuário: Respostas mais completas e úteis

### Logs e Debugging

```python
# Verificar qual modo foi usado na resposta
response = rag_pipeline.query(
    question="Sua pergunta aqui",
    rag_mode="hybrid"
)

print(f"Modo usado: {response['rag_mode']}")
print(f"Provider: {response['provider_used']}")
print(f"Documentos fonte: {len(response['source_documents'])}")
```

## 🔧 Troubleshooting

### Problema: Respostas muito limitadas

**Solução**: Use modo `hybrid` para respostas mais completas

### Problema: Respostas imprecisas

**Solução**: Use modo `strict` para garantir precisão baseada em documentos

### Problema: Cache não funcionando com diferentes modos

**Solução**: O sistema automaticamente inclui o modo na chave do cache

### Problema: Performance lenta com modo hybrid

**Solução**: O modo hybrid pode gerar respostas mais longas; ajuste `max_tokens` se necessário

---

💡 **Recomendação Geral**: Use modo **hybrid** como padrão para a maioria dos casos, e modo **strict** apenas quando precisar de garantias específicas sobre a fonte das informações.
