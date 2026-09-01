FROM python:3.10-slim

# Dependencias del sistema requeridas por UMAP y HDBSCAN
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Ejecutar el procesamiento cuando se inicie el Job
CMD ["python", "app.py"]