FROM python:3.12-slim

WORKDIR /app

# فقط بسته‌های مورد نیاز برای بیلد پکیج‌های پایتون
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      libffi-dev \
      python3-dev \
      libssl-dev && \
    rm -rf /var/lib/apt/lists/*
ENV DB_USER=ds_user
ENV DB_PASS=userpass
ENV DB_HOST=db
ENV DB_NAME=ds_project

# نصب وابستگی‌های پایتون (شامل PyMySQL)
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# کپی کد
COPY pipeline.py ./
COPY scripts/ ./scripts

CMD ["python", "pipeline.py"]
