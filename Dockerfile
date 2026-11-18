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

# Tell Docker that the container listens on port 8080
EXPOSE 8080

# Single worker to avoid shared state issues with global caches - Optimized for 4 cores, 32GB RAM
CMD ["gunicorn", "-w", "1", "--threads", "20", "--worker-class", "gthread", "-b", "0.0.0.0:8080", "--timeout", "60", "app:server"]
