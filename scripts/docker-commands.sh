#!/bin/bash

# =================================================================
# Docker Compose Management Script
# =================================================================
# Utility script for common Docker Compose operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to show usage
show_usage() {
    echo -e "${BLUE}🐳 Haystack + OpenWebUI Docker Management${NC}"
    echo "=============================================="
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  up          - Start all services"
    echo "  down        - Stop all services"
    echo "  restart     - Restart all services"
    echo "  logs        - Show logs for all services"
    echo "  status      - Show status of all services"
    echo "  build       - Build/rebuild services"
    echo "  clean       - Clean up stopped containers and images"
    echo "  validate    - Validate configuration"
    echo "  shell       - Open shell in hayhooks container"
    echo ""
    echo "Examples:"
    echo "  $0 up          # Start all services"
    echo "  $0 logs hayhooks  # Show logs for specific service"
    echo "  $0 shell       # Enter hayhooks container"
}

# Function to validate environment
validate_env() {
    echo -e "${BLUE}🔍 Validating environment...${NC}"
    
    if [ ! -f ".env" ]; then
        echo -e "${RED}❌ Error: .env file not found${NC}"
        echo "Please copy .env.example to .env and configure your API keys"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Environment file found${NC}"
    
    # Run the validation script
    if [ -f "scripts/validate-env.sh" ]; then
        ./scripts/validate-env.sh
    fi
}

# Function to start services
start_services() {
    echo -e "${BLUE}🚀 Starting Haystack + OpenWebUI services...${NC}"
    
    # Validate first
    validate_env
    
    # Create necessary directories
    mkdir -p data/documents data/uploads data/redis open-webui/config
    
    # Start services
    docker compose up -d
    
    echo -e "${GREEN}✅ Services started successfully!${NC}"
    echo ""
    echo "🌐 OpenWebUI: http://localhost:3000"
    echo "🔗 Hayhooks API: http://localhost:1416"
    echo "📊 Redis: localhost:6379"
    echo ""
    echo "Use '$0 logs' to see service logs"
    echo "Use '$0 status' to check service health"
}

# Function to stop services
stop_services() {
    echo -e "${YELLOW}🛑 Stopping Haystack + OpenWebUI services...${NC}"
    docker compose down
    echo -e "${GREEN}✅ Services stopped successfully!${NC}"
}

# Function to restart services
restart_services() {
    echo -e "${BLUE}🔄 Restarting Haystack + OpenWebUI services...${NC}"
    docker compose restart
    echo -e "${GREEN}✅ Services restarted successfully!${NC}"
}

# Function to show logs
show_logs() {
    local service=$2
    if [ -z "$service" ]; then
        echo -e "${BLUE}📋 Showing logs for all services...${NC}"
        docker compose logs -f
    else
        echo -e "${BLUE}📋 Showing logs for $service...${NC}"
        docker compose logs -f "$service"
    fi
}

# Function to show status
show_status() {
    echo -e "${BLUE}📊 Service Status${NC}"
    echo "===================="
    docker compose ps
    echo ""
    echo -e "${BLUE}🔍 Health Checks${NC}"
    echo "===================="
    docker compose exec hayhooks curl -f http://localhost:1416/health 2>/dev/null && echo -e "${GREEN}✅ Hayhooks: Healthy${NC}" || echo -e "${RED}❌ Hayhooks: Unhealthy${NC}"
    docker compose exec open-webui curl -f http://localhost:8080/health 2>/dev/null && echo -e "${GREEN}✅ OpenWebUI: Healthy${NC}" || echo -e "${RED}❌ OpenWebUI: Unhealthy${NC}"
    docker compose exec redis redis-cli ping 2>/dev/null && echo -e "${GREEN}✅ Redis: Healthy${NC}" || echo -e "${RED}❌ Redis: Unhealthy${NC}"
}

# Function to build services
build_services() {
    echo -e "${BLUE}🔨 Building services...${NC}"
    docker compose build --no-cache
    echo -e "${GREEN}✅ Build completed!${NC}"
}

# Function to clean up
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up Docker resources...${NC}"
    docker compose down
    docker system prune -f
    docker volume prune -f
    echo -e "${GREEN}✅ Cleanup completed!${NC}"
}

# Function to open shell
open_shell() {
    local service=${2:-hayhooks}
    echo -e "${BLUE}🐚 Opening shell in $service container...${NC}"
    docker compose exec "$service" /bin/bash
}

# Main script logic
case "${1:-help}" in
    "up"|"start")
        start_services
        ;;
    "down"|"stop")
        stop_services
        ;;
    "restart")
        restart_services
        ;;
    "logs")
        show_logs "$@"
        ;;
    "status"|"ps")
        show_status
        ;;
    "build")
        build_services
        ;;
    "clean"|"cleanup")
        cleanup
        ;;
    "validate"|"config")
        validate_env
        docker compose config
        ;;
    "shell"|"exec")
        open_shell "$@"
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac 