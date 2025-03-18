@echo off
REM Startup script for SurvivAI FastAPI backend on Windows

REM Create necessary directories
mkdir data\datasets 2>nul
mkdir data\models 2>nul
mkdir logs 2>nul

REM Check if .env file exists, if not, copy from template
if not exist .env (
    echo Creating .env file from template...
    copy .env.template .env
    echo Please update the .env file with your configuration settings.
)

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run database migrations
echo Running database migrations...
alembic upgrade head

REM Start the application
echo Starting FastAPI application...
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
