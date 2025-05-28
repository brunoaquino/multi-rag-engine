#!/bin/bash

# =================================================================
# Environment Variables Validation Script
# =================================================================
# This script validates that all required environment variables 
# are properly set in the .env file

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Validating Haystack + OpenWebUI Environment Configuration${NC}"
echo "=================================================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ Error: .env file not found${NC}"
    echo "Please copy .env.example to .env and fill in your API keys:"
    echo "cp .env.example .env"
    exit 1
fi

# Load environment variables
source .env

# Track validation status
ERRORS=0
WARNINGS=0

# Function to check required variable
check_required() {
    local var_name=$1
    local var_value=$2
    local description=$3
    
    if [ -z "$var_value" ]; then
        echo -e "${RED}❌ REQUIRED: $var_name is not set${NC}"
        echo "   Description: $description"
        ((ERRORS++))
    else
        echo -e "${GREEN}✅ $var_name is set${NC}"
    fi
}

# Function to check optional variable
check_optional() {
    local var_name=$1
    local var_value=$2
    local default_value=$3
    
    if [ -z "$var_value" ]; then
        echo -e "${YELLOW}⚠️  OPTIONAL: $var_name is not set (will use default: $default_value)${NC}"
        ((WARNINGS++))
    else
        echo -e "${GREEN}✅ $var_name is set${NC}"
    fi
}

# Function to validate API key format
validate_api_key() {
    local key_name=$1
    local key_value=$2
    local expected_prefix=$3
    
    if [ ! -z "$key_value" ]; then
        if [[ $key_value == $expected_prefix* ]]; then
            echo -e "${GREEN}✅ $key_name format looks correct${NC}"
        else
            echo -e "${YELLOW}⚠️  WARNING: $key_name doesn't start with expected prefix '$expected_prefix'${NC}"
            ((WARNINGS++))
        fi
    fi
}

echo -e "\n${BLUE}🔑 Checking API Keys${NC}"
echo "--------------------------------"
check_required "OPENAI_API_KEY" "$OPENAI_API_KEY" "Required for OpenAI models (GPT-4, embeddings)"
validate_api_key "OPENAI_API_KEY" "$OPENAI_API_KEY" "sk-"

check_required "ANTHROPIC_API_KEY" "$ANTHROPIC_API_KEY" "Required for Anthropic models (Claude)"
validate_api_key "ANTHROPIC_API_KEY" "$ANTHROPIC_API_KEY" "sk-ant-"

check_required "PINECONE_API_KEY" "$PINECONE_API_KEY" "Required for Pinecone vector database"

check_optional "HUGGINGFACE_API_TOKEN" "$HUGGINGFACE_API_TOKEN" "Not set"

echo -e "\n${BLUE}⚙️  Checking Hayhooks Configuration${NC}"
echo "--------------------------------"
check_optional "HAYHOOKS_HOST" "$HAYHOOKS_HOST" "0.0.0.0"
check_optional "HAYHOOKS_PORT" "$HAYHOOKS_PORT" "1416"
check_optional "HAYHOOKS_PIPELINES_DIR" "$HAYHOOKS_PIPELINES_DIR" "/pipelines"

echo -e "\n${BLUE}🌐 Checking OpenWebUI Configuration${NC}"
echo "--------------------------------"
check_optional "OPENAI_API_BASE_URL" "$OPENAI_API_BASE_URL" "http://hayhooks:1416"
check_optional "WEBUI_AUTH" "$WEBUI_AUTH" "false"
check_optional "WEBUI_PORT" "$WEBUI_PORT" "3000"

echo -e "\n${BLUE}🗃️  Checking Pinecone Configuration${NC}"
echo "--------------------------------"
check_required "PINECONE_ENVIRONMENT" "$PINECONE_ENVIRONMENT" "Pinecone environment (e.g., us-east-1-aws)"
check_required "PINECONE_INDEX_NAME" "$PINECONE_INDEX_NAME" "Pinecone index name for storing vectors"
check_optional "PINECONE_DIMENSION" "$PINECONE_DIMENSION" "1536"

echo -e "\n${BLUE}🤖 Checking Model Configuration${NC}"
echo "--------------------------------"
check_optional "DEFAULT_MODEL" "$DEFAULT_MODEL" "gpt-4o-mini"
check_optional "FALLBACK_MODEL" "$FALLBACK_MODEL" "claude-3-5-haiku-20241022"
check_optional "EMBEDDING_MODEL" "$EMBEDDING_MODEL" "text-embedding-3-small"

echo -e "\n${BLUE}📊 Checking Redis Configuration${NC}"
echo "--------------------------------"
check_optional "REDIS_HOST" "$REDIS_HOST" "redis"
check_optional "REDIS_PORT" "$REDIS_PORT" "6379"

echo -e "\n${BLUE}🔒 Checking Security Configuration${NC}"
echo "--------------------------------"
if [ "$AUTH_SECRET" = "your-super-secret-auth-key-change-this" ] || [ "$AUTH_SECRET" = "your-secret-key-here" ]; then
    echo -e "${YELLOW}⚠️  WARNING: AUTH_SECRET is using default value. Please change it!${NC}"
    ((WARNINGS++))
else
    check_optional "AUTH_SECRET" "$AUTH_SECRET" "randomly generated"
fi

echo -e "\n=================================================================="
echo -e "${BLUE}📋 Validation Summary${NC}"
echo "=================================================================="

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ No critical errors found!${NC}"
else
    echo -e "${RED}❌ Found $ERRORS critical error(s) that must be fixed${NC}"
fi

if [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Found $WARNINGS warning(s) - review recommended${NC}"
fi

echo ""
echo "Next steps:"
if [ $ERRORS -eq 0 ]; then
    echo "1. ✅ Environment validation passed"
    echo "2. 🚀 You can now run: docker compose up -d"
    echo "3. 🔍 Check service health: ./scripts/health-check.sh"
else
    echo "1. ❌ Fix the required environment variables above"
    echo "2. 🔄 Run this script again: ./scripts/validate-env.sh"
    echo "3. 📖 See .env.example for reference"
fi

exit $ERRORS 