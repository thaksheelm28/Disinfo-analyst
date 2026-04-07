# Disinformation Analyst — OpenEnv
# Hugging Face Spaces compatible (port 7860)
# docker build -t disinfo-analyst . && docker run -p 7860:7860 disinfo-analyst

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files (exclude venv)
COPY app.py            ./
COPY disinfo_env.py    ./
COPY models.py         ./
COPY grader.py         ./
COPY graph_factory.py  ./
COPY scenarios.json    ./
COPY openenv.yaml      ./
COPY inference.py      ./
COPY pyproject.toml    ./
COPY README.md         ./
COPY tests/            ./tests/

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# HF Spaces requires port 7860
EXPOSE 7860

# Health check — must pass for submission validator
HEALTHCHECK --interval=15s --timeout=10s --start-period=20s --retries=5 \
    CMD curl -sf http://localhost:7860/health || exit 1

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
