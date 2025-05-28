#!/bin/bash
# scripts/check-versions.sh
# Script para verificar versões atuais vs. disponíveis das tecnologias

set -e

echo "🔍 Verificando versões atuais vs. disponíveis..."
echo "=================================================="

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para comparar versões
check_version() {
    local name=$1
    local current=$2
    local latest=$3
    local type=${4:-"info"}
    
    if [ -z "$current" ] || [ -z "$latest" ]; then
        echo -e "${YELLOW}⚠️  $name: Unable to check version${NC}"
        return
    fi
    
    if [ "$current" != "$latest" ]; then
        if [ "$type" = "critical" ]; then
            echo -e "${RED}🔴 $name: $current → $latest (CRITICAL UPDATE)${NC}"
        else
            echo -e "${YELLOW}⚠️  $name: $current → $latest (update available)${NC}"
        fi
        UPDATES_AVAILABLE=true
    else
        echo -e "${GREEN}✅ $name: $current (up to date)${NC}"
    fi
}

# Inicializar flag de updates
UPDATES_AVAILABLE=false

echo -e "\n📦 ${GREEN}Python Dependencies${NC}"
echo "--------------------------------"

# Verificar se requirements.txt existe
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt não encontrado${NC}"
    echo "Criando requirements.txt básico..."
    cat > requirements.txt << 'EOF'
haystack-ai==2.0.0
fastapi==0.104.1
openai==1.3.7
anthropic==0.7.8
streamlit==1.28.1
uvicorn==0.24.0
redis==5.0.1
pinecone-client==2.2.4
pypdf==3.17.0
python-docx==1.1.0
python-multipart==0.0.6
httpx==0.25.2
sentence-transformers==2.2.2
EOF
fi

# Verificar dependências Python principais
echo "Verificando dependências Python..."

# Haystack
if HAYSTACK_CURRENT=$(grep -E "^haystack-ai" requirements.txt | cut -d'=' -f3 2>/dev/null); then
    HAYSTACK_LATEST=$(curl -s https://pypi.org/pypi/haystack-ai/json 2>/dev/null | jq -r .info.version 2>/dev/null || echo "")
    check_version "Haystack AI" "$HAYSTACK_CURRENT" "$HAYSTACK_LATEST" "critical"
else
    echo -e "${YELLOW}⚠️  Haystack AI não encontrado em requirements.txt${NC}"
fi

# FastAPI
if FASTAPI_CURRENT=$(grep -E "^fastapi" requirements.txt | cut -d'=' -f3 2>/dev/null); then
    FASTAPI_LATEST=$(curl -s https://pypi.org/pypi/fastapi/json 2>/dev/null | jq -r .info.version 2>/dev/null || echo "")
    check_version "FastAPI" "$FASTAPI_CURRENT" "$FASTAPI_LATEST"
else
    echo -e "${YELLOW}⚠️  FastAPI não encontrado em requirements.txt${NC}"
fi

# OpenAI
if OPENAI_CURRENT=$(grep -E "^openai" requirements.txt | cut -d'=' -f3 2>/dev/null); then
    OPENAI_LATEST=$(curl -s https://pypi.org/pypi/openai/json 2>/dev/null | jq -r .info.version 2>/dev/null || echo "")
    check_version "OpenAI" "$OPENAI_CURRENT" "$OPENAI_LATEST"
fi

# Anthropic
if ANTHROPIC_CURRENT=$(grep -E "^anthropic" requirements.txt | cut -d'=' -f3 2>/dev/null); then
    ANTHROPIC_LATEST=$(curl -s https://pypi.org/pypi/anthropic/json 2>/dev/null | jq -r .info.version 2>/dev/null || echo "")
    check_version "Anthropic" "$ANTHROPIC_CURRENT" "$ANTHROPIC_LATEST"
fi

# Streamlit
if STREAMLIT_CURRENT=$(grep -E "^streamlit" requirements.txt | cut -d'=' -f3 2>/dev/null); then
    STREAMLIT_LATEST=$(curl -s https://pypi.org/pypi/streamlit/json 2>/dev/null | jq -r .info.version 2>/dev/null || echo "")
    check_version "Streamlit" "$STREAMLIT_CURRENT" "$STREAMLIT_LATEST"
fi

echo -e "\n🐳 ${GREEN}Docker Images${NC}"
echo "--------------------------------"

# Verificar se docker-compose.yml existe
if [ -f "docker-compose.yml" ]; then
    # Redis
    if REDIS_CURRENT=$(grep -A1 "redis:" docker-compose.yml | grep "image:" | cut -d':' -f3 | tr -d ' ' 2>/dev/null); then
        REDIS_LATEST=$(curl -s https://registry.hub.docker.com/v2/repositories/library/redis/tags/?page_size=10 2>/dev/null | jq -r '.results[] | select(.name | test("^[0-9]+\\.[0-9]+$")) | .name' 2>/dev/null | head -1 || echo "")
        check_version "Redis" "$REDIS_CURRENT" "$REDIS_LATEST"
    fi
    
    # Nginx (se usado)
    if NGINX_CURRENT=$(grep -A1 "nginx:" docker-compose.yml | grep "image:" | cut -d':' -f3 | tr -d ' ' 2>/dev/null); then
        NGINX_LATEST="alpine"  # Usando tag alpine como padrão
        check_version "Nginx" "$NGINX_CURRENT" "$NGINX_LATEST"
    fi
else
    echo -e "${YELLOW}⚠️  docker-compose.yml não encontrado${NC}"
fi

echo -e "\n🌐 ${GREEN}GitHub Repositories${NC}"
echo "--------------------------------"

# OpenWebUI
OPENWEBUI_LATEST=$(curl -s https://api.github.com/repos/open-webui/open-webui/releases/latest 2>/dev/null | jq -r .tag_name 2>/dev/null | sed 's/^v//' || echo "")
if [ -n "$OPENWEBUI_LATEST" ]; then
    # Tentar extrair versão atual do docker-compose.yml
    OPENWEBUI_CURRENT=$(grep -A1 "open-webui" docker-compose.yml | grep "image:" | cut -d':' -f3 | tr -d ' ' 2>/dev/null || echo "latest")
    check_version "OpenWebUI" "$OPENWEBUI_CURRENT" "$OPENWEBUI_LATEST"
fi

echo -e "\n🤖 ${GREEN}AI Models${NC}"
echo "--------------------------------"

# Verificar modelos OpenAI (se API key disponível)
if [ -n "$OPENAI_API_KEY" ]; then
    echo "Verificando modelos OpenAI disponíveis..."
    OPENAI_MODELS=$(curl -s -H "Authorization: Bearer $OPENAI_API_KEY" \
        https://api.openai.com/v1/models 2>/dev/null | \
        jq -r '.data[] | select(.id | contains("gpt-4")) | .id' 2>/dev/null | \
        head -3 || echo "")
    
    if [ -n "$OPENAI_MODELS" ]; then
        echo -e "${GREEN}✅ Modelos OpenAI mais recentes:${NC}"
        echo "$OPENAI_MODELS" | while read model; do
            echo "   📱 $model"
        done
    else
        echo -e "${YELLOW}⚠️  Não foi possível verificar modelos OpenAI${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  OPENAI_API_KEY não configurada${NC}"
fi

echo -e "\n📊 ${GREEN}Resumo${NC}"
echo "=================================="

if [ "$UPDATES_AVAILABLE" = true ]; then
    echo -e "${YELLOW}⚠️  Updates disponíveis encontrados!${NC}"
    echo ""
    echo "🔄 Para atualizar as dependências Python:"
    echo "   pip install --upgrade haystack-ai fastapi openai anthropic streamlit"
    echo ""
    echo "🐳 Para atualizar images Docker:"
    echo "   docker-compose pull"
    echo ""
    echo "📝 Lembre-se de atualizar a documentação após os upgrades:"
    echo "   - README.md (seção Tecnologias Utilizadas)"
    echo "   - DEPLOYMENT.md"
    echo "   - requirements.txt"
    
    # Exportar variável para uso em CI/CD
    echo "UPDATES_AVAILABLE=true" >> $GITHUB_ENV 2>/dev/null || true
else
    echo -e "${GREEN}✅ Todas as tecnologias estão atualizadas!${NC}"
    echo "UPDATES_AVAILABLE=false" >> $GITHUB_ENV 2>/dev/null || true
fi

echo -e "\n💡 Para mais informações, consulte:"
echo "   📖 docs/TECHNOLOGY-VERSIONS.md"
echo "   🔧 scripts/update-docs-versions.sh"

exit 0 