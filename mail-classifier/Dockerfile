FROM python:3.11-slim

WORKDIR /app

# Instalamos dependencias del sistema
RUN apt-get update && apt-get install -y git && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Crear carpeta de modelos
RUN mkdir -p /models

# Descargar el modelo de SentenceTransformers en tiempo de build
RUN python - <<EOF
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("/models/all-MiniLM-L6-v2")
EOF

COPY . .

CMD ["python", "main.py"]
