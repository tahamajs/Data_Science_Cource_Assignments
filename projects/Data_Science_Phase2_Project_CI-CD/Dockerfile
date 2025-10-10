FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  build-essential \
  gcc \
  libffi-dev \
  python3-dev \
  libssl-dev \
  default-mysql-client && \
  rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && \
  pip install --no-cache-dir -r ./requirements.txt

# Copy all necessary files
COPY pipeline.py ./
COPY ./scripts/ ./scripts/
COPY ./database/ ./database/
COPY ./src/ ./src/
COPY docker-entrypoint.sh ./

# Make the entrypoint script executable
RUN chmod +x docker-entrypoint.sh

CMD ["./docker-entrypoint.sh"]
