FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for frontend
RUN pip install --no-cache-dir streamlit pillow

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p data/inputs data/outputs data/uploads models/bin tmp visualizations mlruns

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Expose ports for Streamlit
EXPOSE 8501

# Default command (can be overridden)
CMD ["streamlit", "run", "src/frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
