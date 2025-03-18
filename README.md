# SurvivAI Backend - FastAPI Implementation

This repository contains the backend implementation for SurvivAI using FastAPI, providing advanced survival analysis capabilities through a modern API architecture.

## Features

- **Advanced Survival Analysis:** Implementations of Cox Proportional Hazards, Kaplan-Meier, and other survival analysis models
- **RESTful API:** Comprehensive API endpoints for data management and analysis
- **Database Integration:** Async PostgreSQL database with SQLAlchemy for data persistence
- **Model Registry:** Tracking and versioning of trained survival models
- **Visualization Services:** Generation and sharing of survival analysis visualizations
- **Data Processing Pipeline:** Efficient data preprocessing for survival analysis
- **Async Processing:** Non-blocking API design for improved performance

## Project Structure

```
backend/
├── alembic.ini                  # Alembic configuration for database migrations
├── app/
│   ├── main.py                  # Application entry point
│   ├── api/                     # API routes and endpoints
│   │   └── api_v1/              # API version 1
│   │       ├── api.py           # Main API router
│   │       └── endpoints/       # API endpoint modules
│   ├── core/                    # Core application components
│   │   ├── config.py            # Application configuration
│   │   └── dependencies.py      # Dependency injection
│   ├── db/                      # Database related code
│   │   ├── base.py              # Database connection setup
│   │   ├── migrations/          # Alembic migrations
│   │   ├── models/              # SQLAlchemy models
│   │   └── repositories/        # Data access layer
│   ├── ml/                      # Machine learning components
│   │   ├── models/              # Survival analysis model implementations
│   │   └── preprocessing.py     # Data preprocessing utilities
│   ├── models/                  # Pydantic models for request/response validation
│   └── services/                # Business logic layer
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd SurvivAI-NextGen/backend
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables (create a `.env` file in the root directory):

```
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/survivai
API_V1_STR=/api/v1
BACKEND_CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]
DATASET_DIR=./data/datasets
MODEL_DIR=./data/models
```

5. Set up the database:

```bash
# Create the database
# Run migrations using Alembic
alembic upgrade head
```

## Running the Application

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000 and interactive documentation at http://localhost:8000/docs.

## API Documentation

The API documentation is automatically generated and available at:
- Swagger UI: `/docs`
- ReDoc: `/redoc`

## Development

### Database Migrations

When making changes to SQLAlchemy models:

1. Generate a new migration:

```bash
alembic revision --autogenerate -m "description of changes"
```

2. Apply the migration:

```bash
alembic upgrade head
```

### Testing

Run tests with pytest:

```bash
pytest
```

## Key Components

### Models

The backend implements various survival analysis models:

- **Cox Proportional Hazards:** For multivariate survival analysis
- **Kaplan-Meier:** For estimating survival functions

### Analysis Workflow

1. Upload a dataset
2. Create an analysis with specified parameters
3. Run the analysis to fit a model
4. Generate visualizations from the analysis results
5. (Optional) Share visualizations with others

## Frontend Integration

The backend is designed to integrate with the SurvivAI frontend, which can consume the API endpoints to provide a user-friendly interface for survival analysis.
