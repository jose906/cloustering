FROM python:3.10-slim

# Instalar dependencias del sistema requeridas por UMAP y HDBSCAN
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar dependencias e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente (main.py, helper.py, etc.)
COPY . .

# Comando predeterminado (GCP lo sobrescribirá con "python" "main.py")
CMD ["python", "main.py"]