FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data.csv .
COPY config.yaml .
COPY run.py .

CMD ["python", "run.py", \
     "--input",    "data.csv", \
     "--config",   "config.yaml", \
     "--output",   "metrics.json", \
     "--log-file", "run.log"]