FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY . .

RUN mkdir -p /app/data

EXPOSE 8000

RUN chmod +x start.sh

CMD ["sh", "-c", "python init_db.py && python import_faq.py && sh start.sh"]
