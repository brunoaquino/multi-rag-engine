#!/bin/bash

# =================================================================
# Development Environment Startup Script
# =================================================================
# This script provides easy commands to start the development environment
# with appropriate configurations and optional development tools.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${CYAN}=== $1 ===${NC}"
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check for .env file
check_env() {
    if [[ ! -f ".env" ]]; then
        print_warning ".env file not found. Creating from template..."
        if [[ -f ".env.example" ]]; then
            cp .env.example .env
            print_success "Created .env from .env.example"
        else
            print_error ".env.example not found. Please create .env file manually."
            exit 1
        fi
    fi
}

# Function to start basic development environment
start_basic() {
    print_header "Starting Basic Development Environment"
    
    check_docker
    check_env
    
    print_status "Building and starting core services..."
    docker-compose up -d --build
    
    print_success "Basic environment started!"
    show_urls
}

# Function to start development environment with tools
start_with_tools() {
    print_header "Starting Development Environment with Tools"
    
    check_docker
    check_env
    
    print_status "Building and starting all services including development tools..."
    docker-compose --profile dev-tools up -d --build
    
    print_success "Development environment with tools started!"
    show_urls
    show_dev_tools_urls
}

# Function to start in hot-reload mode
start_hot_reload() {
    print_header "Starting Development Environment with Hot-Reload"
    
    check_docker
    check_env
    
    print_status "Starting with hot-reload enabled..."
    
    # Set environment variables for hot-reload
    export RELOAD=true
    export DEBUG=true
    export LOG_LEVEL=DEBUG
    
    docker-compose up -d --build hayhooks
    sleep 5
    docker-compose up -d
    
    print_success "Hot-reload development environment started!"
    print_status "Code changes in ./hayhooks will automatically reload the service"
    show_urls
}

# Function to show service URLs
show_urls() {
    echo
    print_header "Service URLs"
    echo "• OpenWebUI:        http://localhost:3000"
    echo "• Hayhooks API:     http://localhost:8000"
    echo "• Hayhooks Native:  http://localhost:1416"
    echo "• Health Check:     http://localhost:8000/health"
    echo "• API Models:       http://localhost:8000/api/models"
}

# Function to show development tools URLs
show_dev_tools_urls() {
    echo
    print_header "Development Tools"
    echo "• Redis Commander:  http://localhost:8081 (admin/admin)"
    echo "• Adminer:          http://localhost:8082"
    echo "• Mailhog:          http://localhost:8025"
    echo "• Python Debugger:  Port 5678 (connect with debugpy)"
}

# Function to stop all services
stop_all() {
    print_header "Stopping All Services"
    
    print_status "Stopping all containers..."
    docker-compose --profile dev-tools down
    
    print_success "All services stopped!"
}

# Function to clean up development environment
cleanup() {
    print_header "Cleaning Up Development Environment"
    
    print_status "Stopping and removing containers, networks, and volumes..."
    docker-compose --profile dev-tools down -v --remove-orphans
    
    print_status "Removing development images..."
    docker system prune -f
    
    print_success "Cleanup complete!"
}

# Function to show logs
show_logs() {
    local service="${1:-}"
    
    if [[ -n "$service" ]]; then
        print_header "Showing Logs for $service"
        docker-compose logs -f "$service"
    else
        print_header "Showing All Logs"
        docker-compose logs -f
    fi
}

# Function to run health check
health_check() {
    print_header "Running Health Check"
    
    if [[ -f "./scripts/health-check.sh" ]]; then
        ./scripts/health-check.sh
    else
        print_error "Health check script not found at ./scripts/health-check.sh"
        exit 1
    fi
}

# Function to restart a specific service
restart_service() {
    local service="$1"
    
    print_header "Restarting $service"
    
    print_status "Stopping $service..."
    docker-compose stop "$service"
    
    print_status "Starting $service..."
    docker-compose up -d "$service"
    
    print_success "$service restarted!"
}

# Function to rebuild a specific service
rebuild_service() {
    local service="$1"
    
    print_header "Rebuilding $service"
    
    print_status "Stopping $service..."
    docker-compose stop "$service"
    
    print_status "Rebuilding and starting $service..."
    docker-compose up -d --build "$service"
    
    print_success "$service rebuilt and started!"
}

# Function to show help
show_help() {
    echo "Development Environment Control Script"
    echo
    echo "Usage: $0 <command> [options]"
    echo
    echo "Commands:"
    echo "  start              Start basic development environment"
    echo "  start-tools        Start environment with development tools"
    echo "  start-hot          Start with hot-reload enabled"
    echo "  stop               Stop all services"
    echo "  restart <service>  Restart a specific service"
    echo "  rebuild <service>  Rebuild and restart a specific service"
    echo "  logs [service]     Show logs (all services or specific service)"
    echo "  health             Run health check"
    echo "  cleanup            Clean up containers, images, and volumes"
    echo "  urls               Show service URLs"
    echo "  help               Show this help message"
    echo
    echo "Examples:"
    echo "  $0 start                 # Start basic environment"
    echo "  $0 start-tools           # Start with Redis Commander, etc."
    echo "  $0 start-hot             # Start with hot-reload"
    echo "  $0 restart hayhooks      # Restart only Hayhooks service"
    echo "  $0 logs hayhooks         # Show Hayhooks logs"
    echo "  $0 rebuild hayhooks      # Rebuild Hayhooks container"
    echo
    echo "Services: hayhooks, open-webui, redis, redis-commander, adminer, mailhog"
}

# Main command dispatcher
case "${1:-}" in
    start)
        start_basic
        ;;
    start-tools)
        start_with_tools
        ;;
    start-hot)
        start_hot_reload
        ;;
    stop)
        stop_all
        ;;
    restart)
        if [[ -z "${2:-}" ]]; then
            print_error "Please specify a service to restart"
            echo "Available services: hayhooks, open-webui, redis"
            exit 1
        fi
        restart_service "$2"
        ;;
    rebuild)
        if [[ -z "${2:-}" ]]; then
            print_error "Please specify a service to rebuild"
            echo "Available services: hayhooks, open-webui, redis"
            exit 1
        fi
        rebuild_service "$2"
        ;;
    logs)
        show_logs "${2:-}"
        ;;
    health)
        health_check
        ;;
    cleanup)
        cleanup
        ;;
    urls)
        show_urls
        show_dev_tools_urls
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        print_error "No command specified"
        show_help
        exit 1
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac 