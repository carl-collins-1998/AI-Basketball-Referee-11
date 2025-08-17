FROM python:3.9

WORKDIR /app

# Update package sources and install dependencies
RUN echo "deb http://deb.debian.org/debian bullseye main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian-security bullseye-security main contrib non-free" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libgomp1 libgcc-s1 libstdc++6 wget && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . /app/.

# Upgrade pip and install requirements
RUN --mount=type=cache,id=s/6c1ee794-6c5e-4780-8386-2e6f95186369-/root/cache/pip,target=/root/.cache/pip \
    pip install --upgrade pip
RUN --mount=type=cache,id=s/6c1ee794-6c5e-4780-8386-2e6f95186369-/root/cache/pip,target=/root/.cache/pip \
    pip install -r requirements.txt