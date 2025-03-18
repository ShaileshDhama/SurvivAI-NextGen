#!/bin/bash
# Test script for SurvivAI FastAPI backend

# Activate virtual environment
source venv/bin/activate

# Set environment to test
export PYTHONPATH=$PYTHONPATH:$(pwd)
export ENV=test

# Run pytest with coverage
echo "Running tests with coverage..."
pytest --cov=app tests/ -v
