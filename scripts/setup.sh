#!/bin/bash

# =================================================================
# Haystack Project Setup Script
# =================================================================
# This script automates the initial setup process for the Haystack
# project including environment configuration, dependency checks,
# and initial directory structure creation.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project configuration
PROJECT_NAME="Haystack Project"
MIN_DOCKER_VERSION="20.10.0"
MIN_DOCKER_COMPOSE_VERSION="2.0.0"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to compare versions
version_greater_equal() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Function to check Docker version
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker Desktop or Docker Engine."
        print_status "Visit: https://docs.docker.com/get-docker/"
        return 1
    fi
    
    local docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if ! version_greater_equal "$docker_version" "$MIN_DOCKER_VERSION"; then
        print_error "Docker version $docker_version is too old. Minimum required: $MIN_DOCKER_VERSION"
        return 1
    fi
    
    print_success "Docker $docker_version is installed"
    return 0
}

# Function to check Docker Compose version
check_docker_compose() {
    print_status "Checking Docker Compose installation..."
    
    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        print_error "Docker Compose is not installed or available."
        print_status "Install Docker Compose: https://docs.docker.com/compose/install/"
        return 1
    fi
    
    # Try both docker-compose and docker compose commands
    local compose_version=""
    if command_exists docker-compose; then
        compose_version=$(docker-compose --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    elif docker compose version >/dev/null 2>&1; then
        compose_version=$(docker compose version --short 2>/dev/null || docker compose version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    fi
    
    if [[ -z "$compose_version" ]]; then
        print_warning "Could not determine Docker Compose version, but it appears to be installed"
    elif ! version_greater_equal "$compose_version" "$MIN_DOCKER_COMPOSE_VERSION"; then
        print_error "Docker Compose version $compose_version is too old. Minimum required: $MIN_DOCKER_COMPOSE_VERSION"
        return 1
    else
        print_success "Docker Compose $compose_version is installed"
    fi
    
    return 0
}

# Function to check system dependencies
check_dependencies() {
    print_status "Checking system dependencies..."
    
    local missing_deps=()
    
    # Check required commands
    local required_commands=("git" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command_exists "$cmd"; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        print_status "Please install the missing dependencies and run this script again."
        return 1
    fi
    
    print_success "All system dependencies are available"
    return 0
}

# Function to create directory structure
create_directories() {
    print_status "Creating project directory structure..."
    
    local directories=(
        "data/documents"
        "data/uploads"
        "hayhooks/pipelines"
        "open-webui/config"
        "redis"
        "nginx/ssl"
        "scripts"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        fi
    done
    
    print_success "Directory structure created"
}

# Function to setup environment files
setup_environment() {
    print_status "Setting up environment configuration..."
    
    # Check if .env already exists
    if [[ -f ".env" ]]; then
        print_warning ".env file already exists. Creating backup..."
        cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
    fi
    
    # Create .env from template if it doesn't exist
    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.example" ]]; then
            cp .env.example .env
            print_success "Created .env from .env.example"
        else
            print_warning ".env.example not found, creating basic .env file"
            cat > .env << 'EOF'
# =================================================================
# ENVIRONMENT CONFIGURATION
# =================================================================

# AI API Keys
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
PINECONE_API_KEY=

# Services Configuration
WEBUI_PORT=3000
HAYHOOKS_PORT=1416
REDIS_PORT=6379

# Security
WEBUI_SECRET_KEY=your-webui-secret-key
WEBUI_JWT_SECRET_KEY=your-jwt-secret-key

# Features
WEBUI_AUTH=false
ENABLE_SIGNUP=true
ENABLE_RAG=true
EOF
        fi
    fi
    
    # Set appropriate permissions
    chmod 600 .env
    print_success "Environment configuration setup complete"
    
    print_warning "Please edit .env file and add your API keys before running the application"
}

# Function to setup Redis configuration
setup_redis_config() {
    print_status "Setting up Redis configuration..."
    
    if [[ ! -f "redis/redis.conf" ]]; then
        cat > redis/redis.conf << 'EOF'
# Redis configuration for Haystack project
bind 0.0.0.0
port 6379
dir /data
appendonly yes
appendfsync everysec
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
EOF
        print_success "Created Redis configuration"
    else
        print_status "Redis configuration already exists"
    fi
}

# Function to create sample documents
create_sample_documents() {
    print_status "Creating sample documents..."
    
    if [[ ! -f "data/documents/sample.txt" ]]; then
        cat > data/documents/sample.txt << 'EOF'
# Sample Document for Haystack RAG

This is a sample document that demonstrates the RAG (Retrieval-Augmented Generation) capabilities of the Haystack project.

## About Haystack

Haystack is an open-source framework for building search systems that work intelligently over large document collections. It enables developers to implement production-ready neural search, question answering, semantic document search and more.

## Key Features

- Document Search: Find documents using semantic search
- Question Answering: Get precise answers from your documents
- Multi-modal Support: Work with text, images, and other content types
- Production Ready: Built for scale with proper monitoring and deployment tools

## Getting Started

1. Upload your documents to the system
2. The system will automatically process and index them
3. Start asking questions through the chat interface
4. Get intelligent answers based on your document content

This sample document helps demonstrate how the RAG pipeline processes and retrieves information to answer user questions.
EOF
        print_success "Created sample document"
    else
        print_status "Sample document already exists"
    fi
}

# Function to validate setup
validate_setup() {
    print_status "Validating setup..."
    
    local validation_errors=()
    
    # Check if required files exist
    local required_files=("docker-compose.yml" ".env" "hayhooks/Dockerfile")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            validation_errors+=("Missing required file: $file")
        fi
    done
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        validation_errors+=("Docker daemon is not running")
    fi
    
    if [[ ${#validation_errors[@]} -gt 0 ]]; then
        print_error "Validation failed:"
        for error in "${validation_errors[@]}"; do
            print_error "  - $error"
        done
        return 1
    fi
    
    print_success "Setup validation passed"
    return 0
}

# Function to display next steps
show_next_steps() {
    echo
    print_success "Setup completed successfully!"
    echo
    print_status "Next steps:"
    echo "  1. Edit .env file and add your API keys (OpenAI, Anthropic, Pinecone)"
    echo "  2. Run: docker-compose build"
    echo "  3. Run: docker-compose up -d"
    echo "  4. Access OpenWebUI at: http://localhost:3000"
    echo "  5. Run health check: ./scripts/health-check.sh"
    echo
    print_status "For more information, check the documentation in README.md"
}

# Main setup function
main() {
    echo
    print_status "Starting $PROJECT_NAME setup..."
    echo
    
    # Run all checks and setup steps
    check_dependencies || exit 1
    check_docker || exit 1
    check_docker_compose || exit 1
    
    create_directories
    setup_environment
    setup_redis_config
    create_sample_documents
    
    validate_setup || exit 1
    
    show_next_steps
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 