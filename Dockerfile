FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libgcc-s1 \
    libstdc++6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Set environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/best.pt

# Expose the port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
