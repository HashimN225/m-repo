FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git && apt-get clean

# Set working directory
WORKDIR /app

# Copy dependency files first (better layer caching)
COPY requirements.txt .

# Copy setup.py
COPY setup.py .          

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY src/ ./src/

# Install your project as a package
RUN pip install --no-cache-dir .
