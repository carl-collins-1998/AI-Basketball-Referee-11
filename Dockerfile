FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create models directory if it doesn't exist
RUN mkdir -p /app/models

# Download model if MODEL_URL is provided
ARG MODEL_URL
RUN if [ -n "$MODEL_URL" ]; then \
    wget -O /app/models/best.pt "$MODEL_URL"; \
    fi

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]