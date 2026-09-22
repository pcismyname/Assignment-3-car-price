# Build context is the app/ folder (see docker-compose.yaml).
FROM python:3.12-slim

WORKDIR /app

# Install runtime deps first for Docker layer caching.
COPY code/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code + model bundle + the module needed to unpickle it.
COPY code/ .

EXPOSE 8501
# Streamlit must bind 0.0.0.0 to be reachable from outside the container.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
