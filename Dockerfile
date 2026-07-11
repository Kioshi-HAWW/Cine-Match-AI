# Stage 1: Build dependencies and train models
FROM python:3.12-slim as backend-builder
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build essentials for compiling C-extensions (like scikit-surprise)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and dataset needed for training
COPY app ./app
COPY archive ./archive
COPY trained_models ./trained_models

# Train collaborative models (SVD, NMF, KNNBasic) and TF-IDF content vectors during build
RUN python -m app.models.train_collaborative --output-dir /app/trained_models
RUN python -m app.models.train_content --output-dir /app/trained_models

# Delete the massive similarity matrix (759MB) to keep the image optimized.
# The recommender computes this on-the-fly in-memory if the file is missing.
RUN rm -f /app/trained_models/similarity.pkl

# Stage 2: Runtime image
FROM python:3.12-slim as backend
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy python packages and binaries from builder
COPY --from=backend-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy app directory and trained models
COPY --from=backend-builder /app /app

# Run as non-root user for security best practices
RUN useradd --system --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
