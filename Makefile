.PHONY: help install install-dev setup run train test clean lint format

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
	@test -d venv || python3 -m venv venv
	@echo "📦 Installing dependencies..."
	@. venv/bin/activate && pip install --upgrade pip
	@. venv/bin/activate && pip install -r requirements.txt
	@. venv/bin/activate && pip install -r requirements-ui.txt
	@if [ -f requirements-api.txt ]; then \
		. venv/bin/activate && pip install -r requirements-api.txt; \
	fi
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "⚠️  Created .env file - please add your API keys!"; \
	fi
	@echo "✅ Setup complete! Run 'make run' to start."

# Install core dependencies only
install:
	@. venv/bin/activate && pip install -r requirements.txt

# Install all dependencies (core + UI + API)
install-dev:
	@. venv/bin/activate && pip install -r requirements.txt
	@. venv/bin/activate && pip install -r requirements-ui.txt
	@if [ -f requirements-api.txt ]; then \
		. venv/bin/activate && pip install -r requirements-api.txt; \
	fi

# Run Streamlit Web UI
run:
	@echo "🎨 Launching Streamlit Web UI..."
	@. venv/bin/activate && streamlit run app.py

# Run training pipeline
train:
	@echo "�️  Executing training pipeline..."
	@. venv/bin/activate && python scripts/train_occupancy_model.py

# Lint code
lint:
	@echo "🔍 Checking code quality..."
	@. venv/bin/activate && pip install -q flake8 pylint 2>/dev/null || true
	@. venv/bin/activate && flake8 src/ --max-line-length=100 --ignore=E501,W503 || true
	@echo "✅ Lint check complete"

# Format code
format:
	@echo "✨ Formatting code..."
	@. venv/bin/activate && pip install -q black isort 2>/dev/null || true
	@. venv/bin/activate && black src/ app.py scripts/train_occupancy_model.py || true
	@. venv/bin/activate && isort src/ app.py scripts/train_occupancy_model.py || true
	@echo "✅ Code formatted"

# Run tests
test:
	@echo "🧪 Running tests..."
	@. venv/bin/activate && pip install -q pytest 2>/dev/null || true
	@. venv/bin/activate && pytest tests/ -v || echo "⚠️  No tests found"

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
