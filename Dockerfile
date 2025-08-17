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

# Copy entire application including models
COPY . .

# Set environment variables for Ultralytics
ENV YOLO_CONFIG_DIR=/tmp
ENV TMPDIR=/tmp

# Expose the port the app runs on
EXPOSE 8080

# Run the application
CMD ["python", "main.py"]