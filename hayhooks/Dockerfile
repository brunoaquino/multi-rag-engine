# =================================================================
# Hayhooks Dockerfile - Haystack Pipeline Server
# =================================================================
# This Dockerfile extends the official deepset/hayhooks image with
# custom dependencies for OpenAI, Anthropic, Pinecone, and Redis

FROM deepset/hayhooks:v0.8.0

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install additional dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create writable directories for pipelines and data
RUN mkdir -p /app/pipelines /app/data/documents /app/data/uploads /app/logs
RUN chmod -R 755 /app/pipelines /app/data /app/logs

# Copy pipeline files into the container
COPY pipelines/ /app/pipelines/
RUN chmod +x /app/pipelines/*.py

# Copy initialization script, direct API, cache manager, pinecone manager, reranking manager, query processor, load balancer, model manager, and entrypoint
COPY init_pipelines.py /app/
COPY direct_api.py /app/
COPY cache_manager.py /app/
COPY cached_embedder.py /app/
COPY pinecone_manager.py /app/
COPY reranking_manager.py /app/
COPY query_processor.py /app/
COPY load_balancer.py /app/
COPY model_manager.py /app/
COPY entrypoint.sh /app/
RUN chmod +x /app/init_pipelines.py /app/direct_api.py /app/pinecone_manager.py /app/reranking_manager.py /app/query_processor.py /app/load_balancer.py /app/model_manager.py /app/entrypoint.sh

# Set environment variables with defaults
ENV HAYHOOKS_HOST=0.0.0.0
ENV HAYHOOKS_PORT=1416
ENV HAYHOOKS_PIPELINES_DIR=/app/pipelines
ENV HAYHOOKS_MAX_WORKERS=4
ENV HAYHOOKS_RELOAD=true
ENV PYTHONPATH="/app"

# Expose the ports
EXPOSE 1416 8000

# Health check using Python instead of curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:1416/status')" || exit 1

# Default command to run Hayhooks with initialization
CMD ["/app/entrypoint.sh"] 