@echo off
REM Test script for SurvivAI FastAPI backend on Windows

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Set environment to test
set PYTHONPATH=%PYTHONPATH%;%cd%
set ENV=test

REM Run pytest with coverage
echo Running tests with coverage...
pytest --cov=app tests/ -v
