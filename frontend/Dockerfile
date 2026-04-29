FROM python:3.12-slim
WORKDIR /app

# Build this image from the PROJECT ROOT, not from frontend/:
#   docker build -t hashimn/frontend-monitoring:v7 -f frontend/Dockerfile .

COPY frontend/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY frontend/static ./static
COPY frontend/templates ./templates
COPY frontend/app.py .
COPY _feast/ /app/_feast/

EXPOSE 4000
CMD ["python", "app.py"]