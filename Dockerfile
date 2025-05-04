# Usa una imagen base oficial de Python (elige una versión que coincida con tu entorno)
FROM python:3.11-slim

# Establece variables de entorno para evitar diálogos interactivos durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependencias del sistema: Tesseract OCR y utilidades para PDF (necesarias para pdf2image)
# Limpia la caché de apt para reducir el tamaño de la imagen
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requerimientos primero para aprovechar el caché de Docker
COPY requirements.txt .

# Instala las dependencias de Python
# Actualiza pip y usa --no-cache-dir para reducir el tamaño de la capa
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación al directorio de trabajo
COPY . .

# Expone el puerto en el que se ejecutará la aplicación FastAPI (uvicorn por defecto usa 8000)
EXPOSE 8000

# Comando para ejecutar la aplicación usando uvicorn
# Escucha en 0.0.0.0 para ser accesible desde fuera del contenedor
# Asegúrate de que el nombre del archivo sea 'agent.py' y el objeto FastAPI se llame 'app'
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]