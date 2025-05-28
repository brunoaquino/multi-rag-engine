#!/bin/bash
# =================================================================
# Hayhooks Entrypoint Script
# =================================================================

set -e

echo "🚀 Starting Hayhooks with pipeline initialization..."

# Start Hayhooks server in the background
echo "📡 Starting Hayhooks server..."
hayhooks run --host ${HAYHOOKS_HOST} --port ${HAYHOOKS_PORT} --pipelines-dir ${HAYHOOKS_PIPELINES_DIR} &
HAYHOOKS_PID=$!

# Wait a moment for server to start
sleep 5

# Run pipeline initialization in the background
echo "🔧 Initializing pipelines..."
python /app/init_pipelines.py &
INIT_PID=$!

# Start direct API server in the background
echo "🌐 Starting direct chat API on port 8000..."
python /app/direct_api.py &
API_PID=$!

# Function to handle cleanup
cleanup() {
    echo "🛑 Shutting down..."
    kill $HAYHOOKS_PID 2>/dev/null || true
    kill $INIT_PID 2>/dev/null || true
    kill $API_PID 2>/dev/null || true
    wait $HAYHOOKS_PID 2>/dev/null || true
    wait $INIT_PID 2>/dev/null || true
    wait $API_PID 2>/dev/null || true
    exit 0
}

# Set up signal handling
trap cleanup SIGTERM SIGINT

# Wait for Hayhooks server to finish
wait $HAYHOOKS_PID 