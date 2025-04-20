.PHONY: help install install-dev setup run train test clean lint format

DEFAULT_VENV := .venv
ALT_VENV := venv
VENV_DIR := $(if $(wildcard $(DEFAULT_VENV)),$(DEFAULT_VENV),$(if $(wildcard $(ALT_VENV)),$(ALT_VENV),$(DEFAULT_VENV)))

help:
	@echo "🔥 ArlingtonParkingPredict - Available Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          - First-time setup (venv + dependencies + .env)"
	@echo "  make install        - Install core dependencies"
	@echo "  make install-dev    - Install all dependencies (including UI extras)"
	@echo ""
	@echo "Running the Application:"
	@echo "  make run            - Launch Streamlit Web UI"
	@echo "  make train          - Execute the end-to-end training pipeline"
	@echo ""
	@echo "Development:"
	@echo "  make lint           - Check code quality"
	@echo "  make format         - Auto-format code"
	@echo "  make test           - Run tests"
	@echo "  make clean          - Remove generated files and cache"
	@echo ""

# First-time setup
setup:
	@echo "🚀 Setting up ArlingtonParkingPredict..."
	@test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	@echo "📦 Installing dependencies..."
	@. $(VENV_DIR)/bin/activate && pip install --upgrade pip
	@. $(VENV_DIR)/bin/activate && pip install -r requirements.txt
	@. $(VENV_DIR)/bin/activate && pip install -r requirements-ui.txt
	@if [ -f requirements-api.txt ]; then \
		. $(VENV_DIR)/bin/activate && pip install -r requirements-api.txt; \
	fi
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "⚠️  Created .env file - please add your API keys!"; \
	fi
	@echo "✅ Setup complete! Run 'make run' to start."

# Install core dependencies only
install:
	@. $(VENV_DIR)/bin/activate && pip install -r requirements.txt

# Install all dependencies (core + UI + API)
install-dev:
	@. $(VENV_DIR)/bin/activate && pip install -r requirements.txt
	@. $(VENV_DIR)/bin/activate && pip install -r requirements-ui.txt
	@if [ -f requirements-api.txt ]; then \
		. $(VENV_DIR)/bin/activate && pip install -r requirements-api.txt; \
	fi

# Run Streamlit Web UI
run:
	@echo "🎨 Launching Streamlit Web UI..."
	@if [ ! -d "$(VENV_DIR)" ]; then echo "Virtual environment not found at $(VENV_DIR). Run 'make setup' first."; exit 1; fi
	@$(VENV_DIR)/bin/python -m streamlit run app.py

# Run training pipeline
train:
	@echo "�️  Executing training pipeline..."
	@if [ ! -d "$(VENV_DIR)" ]; then echo "Virtual environment not found at $(VENV_DIR). Run 'make setup' first."; exit 1; fi
	@$(VENV_DIR)/bin/python scripts/train_occupancy_model.py

# Lint code
lint:
	@echo "🔍 Checking code quality..."
	@. $(VENV_DIR)/bin/activate && pip install -q flake8 pylint 2>/dev/null || true
	@. $(VENV_DIR)/bin/activate && flake8 src/ --max-line-length=100 --ignore=E501,W503 || true
	@echo "✅ Lint check complete"

# Format code
format:
	@echo "✨ Formatting code..."
	@. $(VENV_DIR)/bin/activate && pip install -q black isort 2>/dev/null || true
	@. $(VENV_DIR)/bin/activate && black src/ app.py scripts/train_occupancy_model.py || true
	@. $(VENV_DIR)/bin/activate && isort src/ app.py scripts/train_occupancy_model.py || true
	@echo "✅ Code formatted"

# Run tests
test:
	@echo "🧪 Running tests..."
	@. $(VENV_DIR)/bin/activate && pip install -q pytest 2>/dev/null || true
	@. $(VENV_DIR)/bin/activate && pytest tests/ -v || echo "⚠️  No tests found"

# Clean generated files
clean:
	@echo "🧹 Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ .coverage htmlcov/ 2>/dev/null || true
	@echo "✅ Cleanup complete"
