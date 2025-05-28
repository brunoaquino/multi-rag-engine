# 🔄 Technology Versions Management Guide

Este documento estabelece o processo para manter as versões das tecnologias documentadas sempre atualizadas.

> 🔗 **Para ver as versões atuais**, consulte a seção "🛠️ Tecnologias Utilizadas" no [README.md](../README.md)

## 📋 Processo de Atualização

### 🗓️ Frequência de Revisão

| Tipo de Tecnologia                     | Frequência      | Responsável   | Documentos Afetados               |
| -------------------------------------- | --------------- | ------------- | --------------------------------- |
| **Core Framework** (Haystack, FastAPI) | Mensalmente     | Tech Lead     | README.md, DEPLOYMENT.md          |
| **AI Models** (OpenAI, Anthropic)      | Semanalmente    | AI Team       | README.md, API.md                 |
| **Infrastructure** (Docker, Nginx)     | Trimestralmente | DevOps        | DEPLOYMENT.md, docker-compose.yml |
| **Python Dependencies**                | Mensalmente     | Dev Team      | requirements.txt, README.md       |
| **Frontend** (OpenWebUI, Streamlit)    | Mensalmente     | Frontend Team | README.md, DEPLOYMENT.md          |

### 🔍 Fontes de Informação

#### **Frameworks Principais**

```bash
# Haystack
curl -s https://api.github.com/repos/deepset-ai/haystack/releases/latest | jq -r .tag_name

# FastAPI
curl -s https://api.github.com/repos/tiangolo/fastapi/releases/latest | jq -r .tag_name

# OpenWebUI
curl -s https://api.github.com/repos/open-webui/open-webui/releases/latest | jq -r .tag_name

# Streamlit
curl -s https://api.github.com/repos/streamlit/streamlit/releases/latest | jq -r .tag_name
```

#### **AI Models e APIs**

```bash
# OpenAI Models (via API)
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models | jq '.data[] | select(.id | contains("gpt")) | .id'

# Verificar novos embeddings
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models | jq '.data[] | select(.id | contains("embedding")) | .id'
```

#### **Docker Images**

```bash
# Verificar tags mais recentes
docker image ls --format "table {{.Repository}}:{{.Tag}}" | grep -E "(redis|nginx|python)"

# Buscar updates
docker pull redis:alpine
docker pull nginx:alpine
docker pull python:3.11-slim
```

### 📝 Checklist de Atualização

#### **1. Verificação Mensal**

- [ ] **Haystack Framework**

  - [ ] Verificar release notes: https://github.com/deepset-ai/haystack/releases
  - [ ] Testar compatibilidade com pipelines existentes
  - [ ] Atualizar README.md e DEPLOYMENT.md

- [ ] **FastAPI**

  - [ ] Verificar changelog: https://github.com/tiangolo/fastapi/releases
  - [ ] Testar APIs após upgrade
  - [ ] Atualizar documentação

- [ ] **OpenWebUI**

  - [ ] Verificar features: https://github.com/open-webui/open-webui/releases
  - [ ] Testar interface de usuário
  - [ ] Atualizar guias de uso

- [ ] **Python Dependencies**
  - [ ] Executar `pip list --outdated`
  - [ ] Verificar breaking changes
  - [ ] Atualizar requirements.txt e documentação

#### **2. Verificação Semanal**

- [ ] **OpenAI Models**

  - [ ] Verificar novos modelos GPT
  - [ ] Testar performance de embeddings
  - [ ] Atualizar configurações padrão

- [ ] **Anthropic Models**
  - [ ] Verificar novos modelos Claude
  - [ ] Testar integração
  - [ ] Atualizar exemplos

#### **3. Verificação Trimestral**

- [ ] **Docker Images**

  - [ ] Redis: Verificar versão LTS
  - [ ] Nginx: Verificar security updates
  - [ ] Python: Verificar versão estável

- [ ] **Infrastructure**
  - [ ] Docker: Verificar compatibilidade
  - [ ] Docker Compose: Verificar features
  - [ ] Kubernetes: Atualizar manifests

## 🛠️ Scripts de Automatização

### Script de Verificação de Versões

```bash
#!/bin/bash
# scripts/check-versions.sh

echo "🔍 Verificando versões atuais vs. disponíveis..."

# Função para comparar versões
check_version() {
    local name=$1
    local current=$2
    local latest=$3

    if [ "$current" != "$latest" ]; then
        echo "⚠️  $name: $current → $latest (update available)"
    else
        echo "✅ $name: $current (up to date)"
    fi
}

# Haystack
HAYSTACK_CURRENT=$(grep "haystack-ai" requirements.txt | cut -d'=' -f3)
HAYSTACK_LATEST=$(curl -s https://pypi.org/pypi/haystack-ai/json | jq -r .info.version)
check_version "Haystack" "$HAYSTACK_CURRENT" "$HAYSTACK_LATEST"

# FastAPI
FASTAPI_CURRENT=$(grep "fastapi" requirements.txt | cut -d'=' -f3)
FASTAPI_LATEST=$(curl -s https://pypi.org/pypi/fastapi/json | jq -r .info.version)
check_version "FastAPI" "$FASTAPI_CURRENT" "$FASTAPI_LATEST"

# OpenAI
OPENAI_CURRENT=$(grep "openai" requirements.txt | cut -d'=' -f3)
OPENAI_LATEST=$(curl -s https://pypi.org/pypi/openai/json | jq -r .info.version)
check_version "OpenAI" "$OPENAI_CURRENT" "$OPENAI_LATEST"
```

### Script de Atualização Automática

```bash
#!/bin/bash
# scripts/update-docs-versions.sh

SOURCE_FILE="README.md"
TEMP_FILE=$(mktemp)

echo "📝 Atualizando versões na documentação..."

# Função para atualizar versão no README
update_doc_version() {
    local package=$1
    local old_version=$2
    local new_version=$3

    sed "s/${package}.*${old_version}/${package}**\` \`v${new_version}+\`/g" "$SOURCE_FILE" > "$TEMP_FILE"
    mv "$TEMP_FILE" "$SOURCE_FILE"
    echo "✅ Atualizado $package: $old_version → $new_version"
}

# Verificar se há mudanças
if git diff --quiet HEAD -- README.md; then
    echo "📝 Documentação atualizada com novas versões"
    echo "💡 Considere revisar e commitar as mudanças"
else
    echo "ℹ️  Nenhuma atualização de versão necessária"
fi
```

## 📊 Dashboard de Versões

### Arquivos de Configuração a Monitorar

| Arquivo              | Propósito                           | Versões Críticas                        |
| -------------------- | ----------------------------------- | --------------------------------------- |
| `requirements.txt`   | Dependencies Python                 | haystack-ai, fastapi, openai, anthropic |
| `package.json`       | Dependencies Node.js (se aplicável) | N/A atualmente                          |
| `docker-compose.yml` | Images Docker                       | redis, nginx, python                    |
| `README.md`          | Documentação principal              | Todas as versões públicas               |
| `DEPLOYMENT.md`      | Guia de deploy                      | Docker, Docker Compose                  |

### Template de Issue para Updates

```markdown
## 🔄 Technology Version Update

**Technology**: [Nome da tecnologia]
**Current Version**: [Versão atual]
**Latest Version**: [Nova versão]
**Breaking Changes**: [Sim/Não]

### 📋 Checklist

- [ ] Review changelog/release notes
- [ ] Test compatibility in development
- [ ] Update requirements.txt (if applicable)
- [ ] Update documentation (README.md, DEPLOYMENT.md, API.md)
- [ ] Update docker-compose.yml (if applicable)
- [ ] Test full deployment
- [ ] Update version tracking spreadsheet

### 🔗 References

- Release Notes: [URL]
- Migration Guide: [URL]
- Breaking Changes: [URL]

### ⚠️ Impact Assessment

**Risk Level**: [Low/Medium/High]
**Components Affected**: [Lista de componentes]
**Rollback Plan**: [Plano de rollback]
```

## 🎯 Automação Futura

### GitHub Actions (Recomendado)

```yaml
# .github/workflows/version-check.yml
name: Technology Version Check

on:
  schedule:
    - cron: "0 9 * * MON" # Every Monday at 9 AM
  workflow_dispatch:

jobs:
  check-versions:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check Versions
        run: ./scripts/check-versions.sh

      - name: Create Issue if Updates Available
        if: env.UPDATES_AVAILABLE == 'true'
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: '🔄 Technology Updates Available',
              body: process.env.UPDATE_SUMMARY,
              labels: ['tech-update', 'documentation']
            })
```

### Dependabot Configuration

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "python"

  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "monthly"
    labels:
      - "dependencies"
      - "docker"
```

## 📚 Recursos de Monitoramento

### APIs e Feeds

- **PyPI RSS**: https://pypi.org/rss/updates.xml
- **GitHub Releases**: Use GitHub API para releases específicos
- **Docker Hub**: APIs para verificar tags
- **OpenAI Updates**: https://platform.openai.com/docs/changelog
- **Anthropic Updates**: https://docs.anthropic.com/claude/changelog

### Ferramentas Recomendadas

- **Renovate**: Automação de dependency updates
- **Snyk**: Security e version monitoring
- **WhiteSource**: License e vulnerability tracking
- **GitHub Dependabot**: Built-in dependency management

---

## 🎯 Próximos Passos

1. **Implementar scripts de verificação** mensalmente
2. **Configurar GitHub Actions** para automação
3. **Estabelecer processo de review** para updates críticos
4. **Documentar breaking changes** e migration paths
5. **Monitorar performance** após updates

---

**📅 Última Atualização**: Janeiro 2024  
**🔄 Próxima Revisão**: Fevereiro 2024  
**👥 Responsável**: Tech Lead + DevOps Team
