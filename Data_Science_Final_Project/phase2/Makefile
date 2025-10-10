.PHONY: help install run clean docker-up docker-down test format lint

# متغیرها
PYTHON := python3
PIP := pip
VENV := venv
DOCKER_COMPOSE := docker-compose -f docker/docker-compose.yml

# رنگ‌ها برای خروجی
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## نمایش این پیام راهنما
	@echo "$(GREEN)دستورات موجود:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""

install: ## نصب وابستگی‌ها
	@echo "$(GREEN)نصب وابستگی‌ها...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✅ نصب با موفقیت انجام شد$(NC)"

venv: ## ایجاد محیط مجازی
	@echo "$(GREEN)ایجاد محیط مجازی...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)✅ محیط مجازی ایجاد شد$(NC)"
	@echo "$(YELLOW)برای فعال‌سازی: source $(VENV)/bin/activate$(NC)"

run: ## اجرای پایپلاین
	@echo "$(GREEN)اجرای پایپلاین...$(NC)"
	$(PYTHON) pipeline.py

jupyter: ## راه‌اندازی Jupyter Notebook
	@echo "$(GREEN)راه‌اندازی Jupyter...$(NC)"
	jupyter notebook notebooks/

lab: ## راه‌اندازی JupyterLab
	@echo "$(GREEN)راه‌اندازی JupyterLab...$(NC)"
	jupyter lab

docker-up: ## راه‌اندازی Docker Compose
	@echo "$(GREEN)راه‌اندازی Docker...$(NC)"
	cd docker && docker-compose up --build

docker-down: ## متوقف کردن Docker Compose
	@echo "$(YELLOW)متوقف کردن Docker...$(NC)"
	cd docker && docker-compose down

docker-clean: ## پاک کردن کامل Docker (همراه volumes)
	@echo "$(RED)پاک کردن کامل Docker...$(NC)"
	cd docker && docker-compose down -v

docker-logs: ## نمایش لاگ‌های Docker
	cd docker && docker-compose logs -f

clean: ## پاک کردن فایل‌های موقت
	@echo "$(YELLOW)پاک کردن فایل‌های موقت...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	@echo "$(GREEN)✅ تمیز شد$(NC)"

clean-output: ## پاک کردن فایل‌های خروجی
	@echo "$(YELLOW)پاک کردن خروجی‌ها...$(NC)"
	rm -rf output/*.csv
	@echo "$(GREEN)✅ خروجی‌ها پاک شدند$(NC)"

test: ## اجرای تست‌ها
	@echo "$(GREEN)اجرای تست‌ها...$(NC)"
	$(PYTHON) -m pytest tests/ -v

format: ## فرمت کردن کد با black
	@echo "$(GREEN)فرمت کردن کد...$(NC)"
	black scripts/ pipeline.py
	@echo "$(GREEN)✅ فرمت شد$(NC)"

lint: ## بررسی کد با flake8
	@echo "$(GREEN)بررسی کد...$(NC)"
	flake8 scripts/ pipeline.py --max-line-length=100
	@echo "$(GREEN)✅ بررسی انجام شد$(NC)"

setup: venv install ## راه‌اندازی کامل پروژه
	@echo "$(GREEN)✅ راه‌اندازی کامل شد$(NC)"
	@echo "$(YELLOW)مرحله بعد: source $(VENV)/bin/activate$(NC)"

mysql-shell: ## ورود به MySQL shell
	mysql -u ds_user -p ds_project

all: clean install run ## اجرای کامل پایپلاین

dev: ## محیط توسعه
	@echo "$(GREEN)راه‌اندازی محیط توسعه...$(NC)"
	$(PIP) install black flake8 pytest ipython
	@echo "$(GREEN)✅ محیط توسعه آماده است$(NC)"

backup: ## بکاپ از داده‌ها
	@echo "$(GREEN)ایجاد بکاپ...$(NC)"
	tar -czf backup_$$(date +%Y%m%d_%H%M%S).tar.gz data/ output/
	@echo "$(GREEN)✅ بکاپ ایجاد شد$(NC)"

requirements: ## بروزرسانی requirements.txt
	@echo "$(GREEN)بروزرسانی requirements...$(NC)"
	$(PIP) freeze > requirements.txt
	@echo "$(GREEN)✅ requirements.txt بروز شد$(NC)"

info: ## نمایش اطلاعات پروژه
	@echo "$(GREEN)═══════════════════════════════════════$(NC)"
	@echo "$(YELLOW)پروژه علم داده - فاز ۲$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════$(NC)"
	@echo "Python: $$(python3 --version)"
	@echo "Pip: $$(pip --version)"
	@echo "Working Directory: $$(pwd)"
	@echo "$(GREEN)═══════════════════════════════════════$(NC)"

