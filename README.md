# SurvivAI-NextGen: Advanced Survival Analysis Platform

<p align="center">
  <img src="https://img.shields.io/github/license/ShaileshDhama/SurvivAI-NextGen" alt="License">
  <img src="https://img.shields.io/github/stars/ShaileshDhama/SurvivAI-NextGen" alt="Stars">
  <img src="https://img.shields.io/github/forks/ShaileshDhama/SurvivAI-NextGen" alt="Forks">
</p>

SurvivAI-NextGen is a comprehensive, modern platform for survival analysis, providing researchers, data scientists, and healthcare professionals with powerful tools for modeling time-to-event data. With an intuitive user interface and robust backend, SurvivAI-NextGen makes advanced survival analysis accessible without sacrificing analytical power.

## 🚀 Features

- **Advanced Survival Analysis Models**: Cox Proportional Hazards, Kaplan-Meier, Deep Learning survival models, and more
- **Interactive Visualizations**: Dynamic survival curves, cumulative hazard plots, and feature importance charts
- **Data Management**: Upload, preprocess, and manage survival datasets
- **Real Estate Market Analysis**: Custom scraper to collect and analyze property listings data
- **Model Registry**: Version and track trained survival models
- **Structured Conversation Exercises**: Language learning through Digital Twin conversation practice
- **API Integration**: Comprehensive RESTful API for programmatic access
- **Cloud-Ready**: Docker and Kubernetes configurations for seamless deployment
- **Type-Safe Frontend**: Modern, responsive UI built with ReScript and React

## 🏗️ Architecture

SurvivAI-NextGen follows a modern, scalable architecture:

- **Backend**: FastAPI with async processing for high-performance API endpoints
- **Frontend**: ReScript/React for type-safe UI development
- **Database**: PostgreSQL with async connectivity via SQLAlchemy
- **Containerization**: Docker for consistent development and production environments
- **Orchestration**: Kubernetes for production deployment

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI
- **Data Processing**: Pandas, NumPy, Polars, DuckDB
- **ML Libraries**: scikit-learn, lifelines, PyTorch, TensorFlow, scikit-survival
- **Database**: PostgreSQL, SQLAlchemy, Alembic
- **Authentication**: JWT, Passlib, Bcrypt

### Frontend
- **Language**: ReScript
- **UI Framework**: React
- **Styling**: Tailwind CSS
- **Charting**: ReCharts
- **Routing**: RescriptReactRouter
- **Build Tool**: Vite

### DevOps
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions

## 📊 Key Applications

### Survival Analysis
1. Upload time-to-event datasets
2. Configure analysis parameters
3. Select appropriate survival models
4. Generate and interpret interactive visualizations
5. Export and share results

### Real Estate Market Analysis
1. Configure scraper tasks for property listings
2. Run scrapers to collect market data
3. Analyze property market trends
4. Visualize property data and pricing patterns

### Digital Twin Language Learning
1. Engage in structured conversation exercises
2. Practice different language skills through role plays
3. Receive grammar and pronunciation feedback
4. Track learning progress over time

## 🔧 Installation

### Prerequisites
- Docker and Docker Compose
- Node.js (v16+)
- Python 3.10+

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/ShaileshDhama/SurvivAI-NextGen.git
cd SurvivAI-NextGen
```

2. Set up environment variables:
```bash
# Copy template files
cp backend/.env.template backend/.env
```

3. Start the development environment:
```bash
docker-compose -f docker-compose.dev.yml up -d
```

### Running Locally (Without Docker)

#### Backend
```bash
cd backend
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 📦 Deployment

### Docker Deployment
```bash
docker-compose up -d
```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/base/
```

## 📚 Documentation

- **API Documentation**: Available at `/docs` when the backend is running
- **Frontend Documentation**: Comprehensive component documentation
- **User Guide**: Detailed instructions for using the platform

## 🧪 Testing

```bash
# Backend Tests
cd backend
python -m pytest

# Frontend Tests
cd frontend
npm test
```

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- The lifelines library for survival analysis
- ReScript and React communities
- FastAPI framework
- All open-source contributors
