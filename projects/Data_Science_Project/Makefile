# Makefile for Data Science Project
# Provides convenient commands for common tasks

.PHONY: help install setup docker-build docker-up docker-down docker-logs clean test lint format

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

##@ General

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\n$(BLUE)Usage:$(NC)\n  make $(YELLOW)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(BLUE)%-15s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(GREEN)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup & Installation

install: ## Install Python dependencies
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed successfully$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest flake8 black mypy isort pre-commit
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

setup-git-lfs: ## Set up Git LFS for large files
	@echo "$(BLUE)Setting up Git LFS...$(NC)"
	git lfs install
	git lfs pull
	@echo "$(GREEN)✓ Git LFS configured$(NC)"

setup-env: ## Create .env file from example
	@echo "$(BLUE)Creating .env file...$(NC)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)✓ .env file created. Please edit it with your settings.$(NC)"; \
	else \
		echo "$(YELLOW)⚠ .env file already exists. Skipping.$(NC)"; \
	fi

setup: install setup-git-lfs setup-env ## Complete initial setup
	@echo "$(GREEN)✓ Project setup complete!$(NC)"

##@ Docker Operations

docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose build
	@echo "$(GREEN)✓ Docker images built$(NC)"

docker-up: ## Start Docker containers
	@echo "$(BLUE)Starting Docker containers...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Containers started$(NC)"
	@echo "$(YELLOW)View logs with: make docker-logs$(NC)"

docker-up-build: ## Build and start Docker containers
	@echo "$(BLUE)Building and starting Docker containers...$(NC)"
	docker-compose up -d --build
	@echo "$(GREEN)✓ Containers built and started$(NC)"

docker-up-tools: ## Start with phpMyAdmin
	@echo "$(BLUE)Starting containers with tools...$(NC)"
	docker-compose --profile tools up -d
	@echo "$(GREEN)✓ Containers started with phpMyAdmin at http://localhost:8080$(NC)"

docker-up-dev: ## Start with development tools (Jupyter)
	@echo "$(BLUE)Starting containers with development tools...$(NC)"
	docker-compose --profile development up -d
	@echo "$(GREEN)✓ Containers started with Jupyter at http://localhost:8888$(NC)"

docker-down: ## Stop Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Containers stopped$(NC)"

docker-down-v: ## Stop containers and remove volumes
	@echo "$(BLUE)Stopping containers and removing volumes...$(NC)"
	docker-compose down -v
	@echo "$(YELLOW)⚠ All data has been removed$(NC)"

docker-restart: ## Restart Docker containers
	@echo "$(BLUE)Restarting Docker containers...$(NC)"
	docker-compose restart
	@echo "$(GREEN)✓ Containers restarted$(NC)"

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-logs-app: ## View application logs only
	docker-compose logs -f app

docker-logs-db: ## View database logs only
	docker-compose logs -f db

docker-ps: ## Show running containers
	docker-compose ps

docker-exec-app: ## Execute bash in app container
	docker-compose exec app bash

docker-exec-db: ## Execute MySQL shell in database container
	docker-compose exec db mysql -u ds_user -puserpass ds_project

##@ Database Operations

db-seed: ## Seed database with CSV data
	@echo "$(BLUE)Seeding database...$(NC)"
	python scripts/seed_database.py
	@echo "$(GREEN)✓ Database seeded$(NC)"

db-schema: ## Import database schema
	@echo "$(BLUE)Importing database schema...$(NC)"
	mysql -u ds_user -p ds_project < database/schema.sql
	@echo "$(GREEN)✓ Schema imported$(NC)"

db-connect: ## Connect to MySQL database
	mysql -u ds_user -p ds_project

db-backup: ## Backup database
	@echo "$(BLUE)Backing up database...$(NC)"
	@mkdir -p backups
	mysqldump -u ds_user -p ds_project > backups/backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✓ Database backed up$(NC)"

##@ Pipeline Operations

pipeline: ## Run the data pipeline
	@echo "$(BLUE)Running data pipeline...$(NC)"
	python pipeline.py
	@echo "$(GREEN)✓ Pipeline completed$(NC)"

load-data: ## Load data from database
	@echo "$(BLUE)Loading data from database...$(NC)"
	python -c "from scripts.load_data import load_data; load_data()"
	@echo "$(GREEN)✓ Data loaded$(NC)"

preprocess: ## Run preprocessing
	@echo "$(BLUE)Running preprocessing...$(NC)"
	python -c "from scripts.preprocess import preprocess_data; from scripts.load_data import load_data; preprocess_data(*load_data())"
	@echo "$(GREEN)✓ Preprocessing completed$(NC)"

##@ Jupyter Operations

jupyter: ## Start Jupyter Lab
	@echo "$(BLUE)Starting Jupyter Lab...$(NC)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

jupyter-notebook: ## Start Jupyter Notebook
	@echo "$(BLUE)Starting Jupyter Notebook...$(NC)"
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

##@ Code Quality

lint: ## Run linting (flake8)
	@echo "$(BLUE)Running flake8...$(NC)"
	flake8 scripts/ src/ --max-line-length=100 --ignore=E203,W503
	@echo "$(GREEN)✓ Linting passed$(NC)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code with black...$(NC)"
	black scripts/ src/
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check: ## Check code formatting
	@echo "$(BLUE)Checking code formatting...$(NC)"
	black --check scripts/ src/

isort: ## Sort imports with isort
	@echo "$(BLUE)Sorting imports...$(NC)"
	isort scripts/ src/
	@echo "$(GREEN)✓ Imports sorted$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy scripts/ src/ --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking passed$(NC)"

quality: lint format type-check ## Run all quality checks
	@echo "$(GREEN)✓ All quality checks passed$(NC)"

##@ Testing

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✓ Tests passed$(NC)"

test-cov: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=scripts --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Tests completed. Coverage report: htmlcov/index.html$(NC)"

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	pytest-watch tests/ -v

##@ Cleaning

clean: ## Clean generated files
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup completed$(NC)"

clean-all: clean docker-down-v ## Clean everything including Docker volumes
	@echo "$(YELLOW)⚠ All data and containers have been removed$(NC)"

clean-output: ## Clean output directories
	@echo "$(BLUE)Cleaning output directories...$(NC)"
	rm -rf output/* logs/* 2>/dev/null || true
	@echo "$(GREEN)✓ Output directories cleaned$(NC)"

##@ Documentation

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "$(YELLOW)Documentation is in README.md$(NC)"

docs-serve: ## Serve documentation (if using mkdocs)
	@echo "$(BLUE)Serving documentation...$(NC)"
	@echo "$(YELLOW)Documentation is in README.md. Use a Markdown viewer.$(NC)"

##@ Development

dev-setup: install-dev setup-env ## Set up development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	pre-commit install
	@echo "$(GREEN)✓ Development environment ready$(NC)"

shell: ## Start Python shell with project context
	@echo "$(BLUE)Starting Python shell...$(NC)"
	python -i -c "from scripts.load_data import load_data; from scripts.preprocess import preprocess_data; from scripts.feature_engineering import engineer_features; print('Functions imported: load_data, preprocess_data, engineer_features')"

##@ Monitoring

status: ## Show project status
	@echo "$(BLUE)=== Project Status ===$(NC)"
	@echo "$(YELLOW)Docker Containers:$(NC)"
	@docker-compose ps 2>/dev/null || echo "Docker not running"
	@echo ""
	@echo "$(YELLOW)Python Version:$(NC)"
	@python --version
	@echo ""
	@echo "$(YELLOW)Database Connection:$(NC)"
	@mysql -u ds_user -puserpass ds_project -e "SELECT COUNT(*) as trip_count FROM uber_trips;" 2>/dev/null || echo "Database not accessible"

watch-logs: ## Watch all logs in real-time
	docker-compose logs -f --tail=100

##@ Quick Commands

quick-start: docker-up-build ## Quick start (build and run)
	@echo "$(GREEN)✓ Project is running!$(NC)"
	@echo "$(YELLOW)Access services at:$(NC)"
	@echo "  MySQL: localhost:3307"
	@echo "  phpMyAdmin: http://localhost:8080 (use --profile tools)"
	@echo "  Jupyter: http://localhost:8888 (use --profile development)"

quick-stop: docker-down ## Quick stop
	@echo "$(GREEN)✓ All services stopped$(NC)"

rebuild: docker-down docker-up-build ## Rebuild and restart
	@echo "$(GREEN)✓ Project rebuilt and restarted$(NC)"

fresh-start: clean-all docker-up-build ## Fresh start (clean and rebuild)
	@echo "$(GREEN)✓ Fresh start completed$(NC)"

##@ Information

version: ## Show versions of tools
	@echo "$(BLUE)=== Tool Versions ===$(NC)"
	@echo "Python: $$(python --version)"
	@echo "Docker: $$(docker --version)"
	@echo "Docker Compose: $$(docker-compose --version)"
	@echo "MySQL: $$(mysql --version)"
	@echo "Git: $$(git --version)"

info: ## Show project information
	@echo "$(BLUE)=== Data Science Project ===$(NC)"
	@echo "Project: Uber Demand Prediction & Analysis"
	@echo "Version: 2.0.0"
	@echo "Python: 3.12+"
	@echo "Database: MySQL 8.0"
	@echo ""
	@echo "$(YELLOW)Quick Commands:$(NC)"
	@echo "  make quick-start    - Start everything"
	@echo "  make docker-logs    - View logs"
	@echo "  make pipeline       - Run pipeline"
	@echo "  make help           - Show all commands"

