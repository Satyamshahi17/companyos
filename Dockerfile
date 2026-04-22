FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY apps/ apps/
COPY env/ env/
COPY server/ server/

# Install Python deps
RUN pip install --no-cache-dir -e .

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Run FastAPI with uvicorn
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]