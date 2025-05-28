# Document Upload Application - Deployment Guide

This guide covers the deployment of the Document Upload Application using Docker and Docker Compose.

## Overview

The application consists of two main components:

- **Streamlit Frontend**: Web interface for document upload and management
- **FastAPI Backend**: RESTful API for document processing and pipeline integration

## Prerequisites

- Docker (version 20.10+)
- Docker Compose (version 2.0+)
- At least 2GB of available RAM
- Ports 8502 and 8503 available

## Quick Start

### 1. Combined Deployment (Recommended)

Deploy both services in a single container:

```bash
./scripts/deploy.sh combined
```

This will:

- Build the Docker image
- Start both Streamlit and FastAPI services
- Create necessary directories
- Perform health checks

Access the application:

- **Streamlit UI**: http://localhost:8502
- **FastAPI Docs**: http://localhost:8503/docs

### 2. Development Mode

For development with live code reloading:

```bash
./scripts/deploy.sh dev
```

This mounts your local code directory into the container for real-time changes.

## Deployment Modes

### Combined Mode (Default)

- Single container running both services
- Managed by Supervisor
- Best for production deployments
- Minimal resource usage

```bash
./scripts/deploy.sh combined
```

### Separate Mode

- Streamlit and FastAPI in separate containers
- Better for scaling individual services
- Useful for microservices architecture

```bash
./scripts/deploy.sh separate
```

### Development Mode

- Volume mounts for live code editing
- Development dependencies included
- Interactive debugging support

```bash
./scripts/deploy.sh dev
```

## Docker Compose Profiles

The application uses Docker Compose profiles for different deployment scenarios:

- **Default**: Combined service
- **separate**: Separate Streamlit and FastAPI services
- **dev**: Development environment
- **proxy**: Nginx reverse proxy (optional)

## Manual Docker Commands

### Build Images

```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build document-upload

# Build without cache
docker-compose build --no-cache
```

### Run Services

```bash
# Combined service
docker-compose up -d document-upload

# Separate services
docker-compose --profile separate up -d

# Development mode
docker-compose --profile dev up -d
```

### View Logs

```bash
# Combined service logs
docker-compose logs -f document-upload

# Separate services logs
docker-compose --profile separate logs -f

# Specific service logs
docker-compose logs -f streamlit
docker-compose logs -f fastapi
```

## Configuration

### Environment Variables

The application supports the following environment variables:

#### Streamlit Configuration

- `STREAMLIT_SERVER_PORT`: Port for Streamlit (default: 8502)
- `STREAMLIT_SERVER_ADDRESS`: Bind address (default: 0.0.0.0)
- `STREAMLIT_SERVER_HEADLESS`: Run in headless mode (default: true)
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS`: Disable usage stats (default: false)

#### FastAPI Configuration

- `FASTAPI_HOST`: Bind address (default: 0.0.0.0)
- `FASTAPI_PORT`: Port for FastAPI (default: 8503)
- `FASTAPI_WORKERS`: Number of worker processes (default: 1)

#### Application Configuration

- `API_BASE_URL`: Base URL for API calls (default: http://localhost:8503)

### Volume Mounts

The application uses the following volumes:

- `./uploads:/app/uploads` - Uploaded files storage
- `./results:/app/results` - Processing results
- `./logs:/var/log/supervisor` - Application logs

## Health Checks

Both services include health check endpoints:

- **Streamlit**: `http://localhost:8502/_stcore/health`
- **FastAPI**: `http://localhost:8503/health`

Health checks run every 30 seconds with a 10-second timeout.

## Scaling

### Horizontal Scaling

For high-traffic deployments, you can scale the services:

```bash
# Scale FastAPI backend
docker-compose --profile separate up -d --scale fastapi=3

# Use load balancer (Nginx) for distribution
docker-compose --profile proxy up -d
```

### Resource Limits

Add resource limits to docker-compose.yml:

```yaml
services:
  document-upload:
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 2G
        reservations:
          cpus: "0.5"
          memory: 512M
```

## Monitoring

### Service Status

Check service status:

```bash
./scripts/deploy.sh status
```

### Logs

View real-time logs:

```bash
# All services
./scripts/deploy.sh combined --logs

# Specific service
docker-compose logs -f document-upload
```

### Metrics

Access application metrics:

- **FastAPI Metrics**: http://localhost:8503/stats
- **Container Stats**: `docker stats`

## Troubleshooting

### Common Issues

#### Port Already in Use

```bash
# Check what's using the port
lsof -i :8502
lsof -i :8503

# Stop conflicting services
./scripts/deploy.sh stop
```

#### Build Failures

```bash
# Clean build
./scripts/deploy.sh build --no-cache

# Check Docker daemon
docker info
```

#### Service Not Starting

```bash
# Check logs
docker-compose logs document-upload

# Check health status
curl http://localhost:8502/_stcore/health
curl http://localhost:8503/health
```

#### Permission Issues

```bash
# Fix directory permissions
sudo chown -R $USER:$USER uploads results logs
```

### Debug Mode

Run in debug mode for troubleshooting:

```bash
# Interactive shell in container
docker-compose --profile dev run --rm dev /bin/bash

# Run specific commands
docker-compose --profile dev run --rm dev python test_api_server.py
```

## Security Considerations

### Production Deployment

For production deployments:

1. **Use HTTPS**: Configure SSL certificates
2. **Firewall**: Restrict access to necessary ports
3. **Authentication**: Implement user authentication
4. **File Validation**: Ensure robust file validation
5. **Resource Limits**: Set appropriate resource limits

### SSL Configuration

Add SSL support with Nginx proxy:

```bash
# Generate SSL certificates
mkdir -p docker/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/ssl/nginx.key \
  -out docker/ssl/nginx.crt

# Deploy with proxy
docker-compose --profile proxy up -d
```

## Backup and Recovery

### Data Backup

```bash
# Backup uploads and results
tar -czf backup-$(date +%Y%m%d).tar.gz uploads/ results/

# Backup to remote location
rsync -av uploads/ results/ user@backup-server:/path/to/backup/
```

### Database Backup (if using external DB)

```bash
# PostgreSQL example
docker exec postgres-container pg_dump -U user dbname > backup.sql

# MongoDB example
docker exec mongo-container mongodump --out /backup
```

## Performance Optimization

### Image Optimization

- Use multi-stage builds (already implemented)
- Minimize layer count
- Use .dockerignore effectively
- Choose appropriate base images

### Runtime Optimization

- Adjust worker processes based on CPU cores
- Configure memory limits appropriately
- Use volume mounts for large files
- Implement caching strategies

## Maintenance

### Updates

```bash
# Pull latest images
./scripts/deploy.sh build --pull

# Restart services
./scripts/deploy.sh stop
./scripts/deploy.sh combined
```

### Cleanup

```bash
# Remove unused containers and images
./scripts/deploy.sh clean

# Prune Docker system
docker system prune -a
```

## Support

For issues and questions:

1. Check the logs: `./scripts/deploy.sh status`
2. Review this documentation
3. Check Docker and Docker Compose versions
4. Verify system requirements

## Script Reference

The deployment script (`./scripts/deploy.sh`) supports the following commands:

- `combined` - Deploy combined service (default)
- `separate` - Deploy separate services
- `dev` - Deploy in development mode
- `build` - Build images only
- `test` - Run tests in container
- `stop` - Stop all services
- `clean` - Clean up containers and images
- `status` - Show service status

Options:

- `--no-cache` - Build without cache
- `--pull` - Pull latest base images
- `--logs` - Show logs after deployment
- `--help` - Show help message
