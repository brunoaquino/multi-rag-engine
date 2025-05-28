#!/bin/bash

# Document Upload Application Deployment Script
# Usage: ./scripts/deploy.sh [mode] [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="combined"
BUILD_ARGS=""
COMPOSE_ARGS=""

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

# Function to show usage
show_usage() {
    echo "Document Upload Application Deployment Script"
    echo ""
    echo "Usage: $0 [mode] [options]"
    echo ""
    echo "Modes:"
    echo "  combined    - Deploy both Streamlit and FastAPI in one container (default)"
    echo "  separate    - Deploy Streamlit and FastAPI in separate containers"
    echo "  dev         - Deploy in development mode with volume mounts"
    echo "  build       - Build Docker images only"
    echo "  test        - Run tests in container"
    echo "  stop        - Stop all services"
    echo "  clean       - Stop and remove all containers and images"
    echo ""
    echo "Options:"
    echo "  --no-cache  - Build without Docker cache"
    echo "  --pull      - Pull latest base images"
    echo "  --logs      - Show logs after deployment"
    echo "  --help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 combined                    # Deploy combined service"
    echo "  $0 separate --logs             # Deploy separate services and show logs"
    echo "  $0 dev                         # Deploy in development mode"
    echo "  $0 build --no-cache            # Build images without cache"
    echo "  $0 test                        # Run tests"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p uploads results logs
    print_success "Directories created"
}

# Function to build images
build_images() {
    print_status "Building Docker images..."
    
    if [[ "$BUILD_ARGS" == *"--no-cache"* ]]; then
        COMPOSE_ARGS="$COMPOSE_ARGS --no-cache"
    fi
    
    if [[ "$BUILD_ARGS" == *"--pull"* ]]; then
        COMPOSE_ARGS="$COMPOSE_ARGS --pull"
    fi
    
    case $MODE in
        "combined")
            docker-compose build $COMPOSE_ARGS document-upload
            ;;
        "separate")
            docker-compose --profile separate build $COMPOSE_ARGS
            ;;
        "dev")
            docker-compose --profile dev build $COMPOSE_ARGS
            ;;
        "build")
            docker-compose build $COMPOSE_ARGS
            docker-compose --profile separate build $COMPOSE_ARGS
            docker-compose --profile dev build $COMPOSE_ARGS
            ;;
    esac
    
    print_success "Docker images built successfully"
}

# Function to deploy services
deploy_services() {
    print_status "Deploying services in $MODE mode..."
    
    case $MODE in
        "combined")
            docker-compose up -d document-upload
            ;;
        "separate")
            docker-compose --profile separate up -d
            ;;
        "dev")
            docker-compose --profile dev up -d
            ;;
    esac
    
    print_success "Services deployed successfully"
}

# Function to run tests
run_tests() {
    print_status "Running tests in container..."
    
    # Build test image
    docker-compose --profile dev build dev
    
    # Run tests
    docker-compose --profile dev run --rm dev python -m pytest -v
    
    print_success "Tests completed"
}

# Function to show logs
show_logs() {
    print_status "Showing service logs..."
    
    case $MODE in
        "combined")
            docker-compose logs -f document-upload
            ;;
        "separate")
            docker-compose --profile separate logs -f
            ;;
        "dev")
            docker-compose --profile dev logs -f
            ;;
    esac
}

# Function to stop services
stop_services() {
    print_status "Stopping all services..."
    
    docker-compose down
    docker-compose --profile separate down
    docker-compose --profile dev down
    docker-compose --profile proxy down
    
    print_success "All services stopped"
}

# Function to clean up
clean_up() {
    print_status "Cleaning up containers and images..."
    
    # Stop all services
    stop_services
    
    # Remove containers
    docker-compose down --remove-orphans
    docker-compose --profile separate down --remove-orphans
    docker-compose --profile dev down --remove-orphans
    docker-compose --profile proxy down --remove-orphans
    
    # Remove images
    docker images | grep document-upload | awk '{print $3}' | xargs -r docker rmi -f
    
    # Remove volumes (optional)
    read -p "Do you want to remove volumes (uploads, results, logs)? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down --volumes
        print_success "Volumes removed"
    fi
    
    print_success "Cleanup completed"
}

# Function to show status
show_status() {
    print_status "Service status:"
    echo ""
    docker-compose ps
    echo ""
    
    print_status "Health checks:"
    
    # Check Streamlit
    if curl -f http://localhost:8502/_stcore/health &> /dev/null; then
        print_success "Streamlit is healthy (http://localhost:8502)"
    else
        print_warning "Streamlit is not responding"
    fi
    
    # Check FastAPI
    if curl -f http://localhost:8503/health &> /dev/null; then
        print_success "FastAPI is healthy (http://localhost:8503)"
    else
        print_warning "FastAPI is not responding"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        combined|separate|dev|build|test|stop|clean|status)
            MODE="$1"
            shift
            ;;
        --no-cache)
            BUILD_ARGS="$BUILD_ARGS --no-cache"
            shift
            ;;
        --pull)
            BUILD_ARGS="$BUILD_ARGS --pull"
            shift
            ;;
        --logs)
            SHOW_LOGS=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "Starting deployment script for Document Upload Application"
    print_status "Mode: $MODE"
    
    case $MODE in
        "stop")
            stop_services
            ;;
        "clean")
            clean_up
            ;;
        "status")
            show_status
            ;;
        "test")
            check_prerequisites
            run_tests
            ;;
        "build")
            check_prerequisites
            create_directories
            build_images
            ;;
        *)
            check_prerequisites
            create_directories
            build_images
            deploy_services
            
            # Wait a moment for services to start
            sleep 5
            
            show_status
            
            if [[ "$SHOW_LOGS" == "true" ]]; then
                show_logs
            else
                print_success "Deployment completed!"
                print_status "Access the application:"
                print_status "  Streamlit UI: http://localhost:8502"
                print_status "  FastAPI Docs: http://localhost:8503/docs"
                print_status ""
                print_status "To view logs: $0 $MODE --logs"
                print_status "To stop services: $0 stop"
            fi
            ;;
    esac
}

# Run main function
main 