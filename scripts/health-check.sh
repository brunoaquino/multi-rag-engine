#!/bin/bash

# =================================================================
# Haystack Project Health Check Script
# =================================================================
# This script performs comprehensive health checks on all components
# of the system to verify operational status and proper configuration.

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
HAYHOOKS_URL="http://localhost:8000"
HAYHOOKS_API_URL="http://localhost:1416"
WEBUI_URL="http://localhost:3000"
REDIS_URL="localhost:6379"
TIMEOUT=10

# Health check counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

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

print_checking() {
    echo -e "${BLUE}[→]${NC} Checking $1..."
}

# Function to record check result
record_result() {
    local status="$1"
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    case "$status" in
        "PASS") PASSED_CHECKS=$((PASSED_CHECKS + 1)) ;;
        "FAIL") FAILED_CHECKS=$((FAILED_CHECKS + 1)) ;;
        "WARN") WARNING_CHECKS=$((WARNING_CHECKS + 1)) ;;
    esac
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to make HTTP request with timeout
http_check() {
    local url="$1"
    local expected_code="${2:-200}"
    local timeout="${3:-$TIMEOUT}"
    
    if command_exists curl; then
        response=$(curl -s -w "%{http_code}" -o /dev/null -m "$timeout" "$url" 2>/dev/null || echo "000")
        [[ "$response" =~ $expected_code ]]
    else
        return 1
    fi
}

# Function to check JSON response
json_check() {
    local url="$1"
    local timeout="${2:-$TIMEOUT}"
    
    if command_exists curl && command_exists jq; then
        response=$(curl -s -m "$timeout" "$url" 2>/dev/null | jq . 2>/dev/null)
        [[ -n "$response" ]]
    else
        return 1
    fi
}

# Function to check Docker containers
check_docker_containers() {
    print_header "Docker Containers Health"
    
    if ! command_exists docker; then
        print_error "Docker is not installed"
        record_result "FAIL"
        return 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running"
        record_result "FAIL"
        return 1
    fi
    
    # Check if containers are running
    local containers=("hayhooks" "open-webui" "redis")
    local container_status=0
    
    for container in "${containers[@]}"; do
        print_checking "$container container"
        if docker ps --format "table {{.Names}}" | grep -q "^$container$"; then
            local status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "unknown")
            case "$status" in
                "healthy")
                    print_success "$container is running and healthy"
                    ;;
                "unhealthy")
                    print_error "$container is running but unhealthy"
                    container_status=1
                    ;;
                "starting")
                    print_warning "$container is starting up"
                    ;;
                *)
                    local state=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null || echo "unknown")
                    if [[ "$state" == "running" ]]; then
                        print_success "$container is running (no health check configured)"
                    else
                        print_error "$container is not running (status: $state)"
                        container_status=1
                    fi
                    ;;
            esac
        else
            print_error "$container container is not running"
            container_status=1
        fi
    done
    
    if [[ $container_status -eq 0 ]]; then
        record_result "PASS"
    else
        record_result "FAIL"
    fi
    
    return $container_status
}

# Function to check Hayhooks service
check_hayhooks() {
    print_header "Hayhooks Service Health"
    
    local hayhooks_status=0
    
    # Check main API endpoint
    print_checking "Hayhooks main API ($HAYHOOKS_URL)"
    if http_check "$HAYHOOKS_URL/health"; then
        print_success "Hayhooks main API is responding"
        
        # Check if we can get health details
        if json_check "$HAYHOOKS_URL/health"; then
            local health_data=$(curl -s "$HAYHOOKS_URL/health" 2>/dev/null)
            print_status "Health details: $health_data"
        fi
    else
        print_error "Hayhooks main API is not responding"
        hayhooks_status=1
    fi
    
    # Check Hayhooks native API
    print_checking "Hayhooks native API ($HAYHOOKS_API_URL)"
    if http_check "$HAYHOOKS_API_URL/status"; then
        print_success "Hayhooks native API is responding"
        
        # Check if we can get status details
        if json_check "$HAYHOOKS_API_URL/status"; then
            local status_data=$(curl -s "$HAYHOOKS_API_URL/status" 2>/dev/null)
            print_status "Status details: $status_data"
        fi
    else
        print_error "Hayhooks native API is not responding"
        hayhooks_status=1
    fi
    
    # Check OpenAI-compatible endpoints
    print_checking "OpenAI-compatible API endpoints"
    if http_check "$HAYHOOKS_URL/api/models"; then
        print_success "Models endpoint is available"
        
        # Check available models
        if json_check "$HAYHOOKS_URL/api/models"; then
            local models=$(curl -s "$HAYHOOKS_URL/api/models" 2>/dev/null | jq -r '.data[].id' 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
            print_status "Available models: $models"
        fi
    else
        print_error "OpenAI-compatible endpoints not available"
        hayhooks_status=1
    fi
    
    if [[ $hayhooks_status -eq 0 ]]; then
        record_result "PASS"
    else
        record_result "FAIL"
    fi
}

# Function to check OpenWebUI
check_openwebui() {
    print_header "OpenWebUI Health"
    
    print_checking "OpenWebUI interface ($WEBUI_URL)"
    if http_check "$WEBUI_URL" "200|302"; then
        print_success "OpenWebUI is accessible"
        record_result "PASS"
    else
        print_error "OpenWebUI is not accessible"
        record_result "FAIL"
    fi
    
    # Check health endpoint if available
    print_checking "OpenWebUI health endpoint"
    if http_check "$WEBUI_URL/health"; then
        print_success "OpenWebUI health endpoint is responding"
    else
        print_warning "OpenWebUI health endpoint not available (this may be normal)"
    fi
}

# Function to check Redis
check_redis() {
    print_header "Redis Health"
    
    print_checking "Redis connection"
    if command_exists redis-cli; then
        if redis-cli -h localhost -p 6379 ping >/dev/null 2>&1; then
            print_success "Redis is responding to ping"
            
            # Get Redis info
            local redis_info=$(redis-cli -h localhost -p 6379 info server 2>/dev/null | grep "redis_version" | cut -d: -f2 | tr -d '\r\n')
            if [[ -n "$redis_info" ]]; then
                print_status "Redis version: $redis_info"
            fi
            
            record_result "PASS"
        else
            print_error "Redis is not responding"
            record_result "FAIL"
        fi
    else
        # Try alternative check using Docker
        if docker exec redis redis-cli ping >/dev/null 2>&1; then
            print_success "Redis is responding (via Docker)"
            record_result "PASS"
        else
            print_error "Cannot check Redis (redis-cli not available and Docker check failed)"
            record_result "FAIL"
        fi
    fi
}

# Function to check environment configuration
check_environment() {
    print_header "Environment Configuration"
    
    # Check if .env file exists
    print_checking ".env file"
    if [[ -f ".env" ]]; then
        print_success ".env file exists"
        
        # Check for required API keys (without exposing values)
        local required_vars=("OPENAI_API_KEY" "ANTHROPIC_API_KEY" "PINECONE_API_KEY")
        local missing_vars=()
        
        for var in "${required_vars[@]}"; do
            if ! grep -q "^$var=" .env || grep -q "^$var=$" .env || grep -q "^$var=your-" .env; then
                missing_vars+=("$var")
            fi
        done
        
        if [[ ${#missing_vars[@]} -eq 0 ]]; then
            print_success "All required API keys are configured"
            record_result "PASS"
        else
            print_warning "Missing or placeholder API keys: ${missing_vars[*]}"
            record_result "WARN"
        fi
    else
        print_error ".env file not found"
        record_result "FAIL"
    fi
}

# Function to check disk space
check_resources() {
    print_header "System Resources"
    
    # Check disk space
    print_checking "Disk space"
    local available_space=$(df . | awk 'NR==2 {print $4}')
    local available_gb=$((available_space / 1024 / 1024))
    
    if [[ $available_gb -gt 5 ]]; then
        print_success "Sufficient disk space available: ${available_gb}GB"
        record_result "PASS"
    elif [[ $available_gb -gt 1 ]]; then
        print_warning "Low disk space: ${available_gb}GB available"
        record_result "WARN"
    else
        print_error "Very low disk space: ${available_gb}GB available"
        record_result "FAIL"
    fi
    
    # Check if Docker has enough space
    if command_exists docker; then
        print_checking "Docker system resources"
        if docker system df >/dev/null 2>&1; then
            print_success "Docker system is healthy"
        else
            print_warning "Could not check Docker system resources"
        fi
    fi
}

# Function to run API functionality test
test_api_functionality() {
    print_header "API Functionality Test"
    
    # Test chat completion endpoint
    print_checking "Chat completion functionality"
    local test_payload='{"messages":[{"role":"user","content":"Hello, test message"}],"model":"haystack-chat","max_tokens":50}'
    
    if command_exists curl && command_exists jq; then
        local response=$(curl -s -w "%{http_code}" -X POST \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer haystack-api-key" \
            -d "$test_payload" \
            -m 30 \
            "$HAYHOOKS_URL/api/chat/completions" 2>/dev/null)
        
        local http_code="${response: -3}"
        if [[ "$http_code" == "200" ]]; then
            print_success "Chat completion API is functional"
            record_result "PASS"
        else
            print_error "Chat completion API returned HTTP $http_code"
            record_result "FAIL"
        fi
    else
        print_warning "Cannot test API functionality (curl or jq not available)"
        # Don't count this as a check since we can't run it
    fi
}

# Function to display summary
display_summary() {
    print_header "Health Check Summary"
    
    echo
    print_status "Results: $PASSED_CHECKS passed, $FAILED_CHECKS failed, $WARNING_CHECKS warnings out of $TOTAL_CHECKS checks"
    
    if [[ $FAILED_CHECKS -eq 0 ]]; then
        if [[ $WARNING_CHECKS -eq 0 ]]; then
            print_success "All systems are healthy! 🎉"
            return 0
        else
            print_warning "System is mostly healthy with some warnings ⚠️"
            return 0
        fi
    else
        print_error "Some critical issues found! Please check the failed components ❌"
        return 1
    fi
}

# Main function
main() {
    echo
    print_header "Haystack Project Health Check"
    echo
    
    # Run all health checks
    check_docker_containers
    check_hayhooks
    check_openwebui
    check_redis
    check_environment
    check_resources
    test_api_functionality
    
    echo
    display_summary
}

# Show help
show_help() {
    echo "Haystack Project Health Check Script"
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --verbose  Show verbose output"
    echo "  -q, --quick    Run quick checks only (skip API tests)"
    echo
    echo "This script checks the health of all Haystack project components:"
    echo "  • Docker containers status"
    echo "  • Hayhooks API endpoints"
    echo "  • OpenWebUI accessibility"
    echo "  • Redis connectivity"
    echo "  • Environment configuration"
    echo "  • System resources"
    echo "  • API functionality"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    -v|--verbose)
        set -x
        main
        ;;
    -q|--quick)
        echo "Quick health check mode (skipping API tests)"
        # Redefine test_api_functionality to skip
        test_api_functionality() {
            print_header "API Functionality Test"
            print_status "Skipped in quick mode"
        }
        main
        ;;
    *)
        main
        ;;
esac 