# ============================================================
# Dockerfile — Observable RAG Platform
# ============================================================
# Base: Python 3.12 slim (minimal attack surface)
# Runs: uvicorn on port 8000
# ============================================================

FROM python:3.12-slim

# System dependencies needed by pymupdf and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure the logs directory exists inside the container
RUN mkdir -p logs

# Expose FastAPI port
EXPOSE 8000

# Health-check so docker-compose knows when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run with uvicorn
CMD ["uvicorn", "rag_app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
