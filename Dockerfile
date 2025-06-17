 # Use Python 3.10 as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY app ./app
COPY data/models ./data/models
COPY data/media ./data/media

# Install dependencies using uv
RUN uv sync --frozen --no-dev --no-editable

# Expose port
EXPOSE 7860

# Set the entry point
CMD ["uv", "run", "app/main.py"]