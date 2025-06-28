.PHONY: help install test lint format clean docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  test         Run unit tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean build artifacts"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run training in Docker"

install:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

test:
	pytest tests/ -v --cov=janus.training.ppo --cov-report=term-missing

lint:
	flake8 janus/training/ppo --max-line-length=88
	mypy janus/training/ppo --ignore-missing-imports
	black --check janus/training/ppo
	isort --check-only janus/training/ppo

format:
	black janus/training/ppo
	isort janus/training/ppo

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info
	rm -rf .coverage .pytest_cache .mypy_cache

docker-build:
	docker build -t janus-ppo:latest .

docker-run:
	docker-compose up -d
