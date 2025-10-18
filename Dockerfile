
FROM python:3.11-slim

# Instala tesseract
RUN apt-get update && apt-get install -y --no-install-recommends     tesseract-ocr  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Instala CLI + OCRs CPU
RUN pip install --no-cache-dir -e ".[all-cpu]"

ENTRYPOINT ["daa"]
CMD ["version"]
