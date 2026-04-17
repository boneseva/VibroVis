# Use a stable Debian-based Python image
FROM python:3.11-slim-bookworm

# Ensure output logs are sent directly to terminal
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Create the cache directory with proper permissions
RUN mkdir -p /app/cache/manual_labels && \
    chmod 777 /app/cache/manual_labels

# Tell Docker that the container listens on port 8080
EXPOSE 8080

# Enhanced multi-worker configuration:
# - Multiple workers for better performance
# - Shared cache directory for label persistence
# - Increased timeout for long-running operations
# - Worker restart for memory management
CMD ["gunicorn", \
     "--workers", "3", \
     "--threads", "2", \
     "--worker-class", "sync", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--preload", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--log-level", "info", \
     "app:server"]
