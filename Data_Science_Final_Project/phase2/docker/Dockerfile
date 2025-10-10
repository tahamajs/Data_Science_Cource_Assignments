FROM python:3.12-slim

WORKDIR /app

# نصب بسته‌های مورد نیاز برای بیلد پکیج‌های پایتون
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  build-essential \
  gcc \
  libffi-dev \
  python3-dev \
  libssl-dev && \
  rm -rf /var/lib/apt/lists/*

# کپی فایل requirements و نصب وابستگی‌ها
COPY requirements.txt ./
RUN pip install --upgrade pip && \
  pip install --no-cache-dir -r ./requirements.txt && \
  pip install cryptography

# کپی فایل‌های پروژه
COPY pipeline.py ./
COPY scripts/ ./scripts/

# اجرای pipeline
CMD ["python", "pipeline.py"]
