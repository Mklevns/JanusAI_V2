FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command
CMD ["python", "janus/training/ppo/main.py"]
