# SurvivAI Frontend - ReScript Implementation

This repository contains the frontend implementation for SurvivAI using ReScript and React, providing a modern, type-safe UI for survival analysis.

## Tech Stack

- **Language**: [ReScript](https://rescript-lang.org/) - A robustly typed language that compiles to efficient JavaScript
- **UI Library**: [React](https://reactjs.org/) with ReScript bindings
- **Styling**: [Tailwind CSS](https://tailwindcss.com/) for utility-first styling
- **State Management**: ReScript's built-in immutable data structures with React Context
- **Data Fetching**: Custom fetch hooks with ReScript's Promise API
- **Visualization**: [ReCharts](https://recharts.org/) with ReScript bindings
- **Routing**: [RescriptReactRouter](https://rescript-lang.org/docs/react/latest/router)
- **Build Tool**: [Vite](https://vitejs.dev/) for fast development and optimized production builds

## Project Structure

```
frontend/
├── rescript.json           # ReScript configuration
├── package.json            # NPM dependencies
├── vite.config.js          # Vite build configuration
├── tailwind.config.js      # Tailwind CSS configuration
├── postcss.config.js       # PostCSS configuration
├── src/
│   ├── App.res             # Main application component
│   ├── Index.res           # Entry point
│   ├── bindings/           # ReScript bindings to JS libraries
│   ├── components/         # Reusable UI components
│   │   ├── layout/         # Layout components
│   │   ├── visualizations/ # Visualization components
│   │   ├── analyses/       # Analysis components
│   │   └── common/         # Common UI components
│   ├── pages/              # Page components
│   ├── contexts/           # React contexts for state management
│   ├── services/           # API service clients
│   ├── hooks/              # Custom React hooks
│   ├── types/              # ReScript type definitions
│   └── utils/              # Utility functions
├── public/                 # Static assets
└── styles/                 # Global styles
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd SurvivAI-NextGen/frontend
```

2. Install dependencies:

```bash
npm install
```

3. Set up environment variables (create a `.env` file in the root directory):

```
VITE_API_URL=http://localhost:8000/api/v1
```

## Development

Start the development server:

```bash
npm run dev
```

The application will be available at http://localhost:5173.

## Building for Production

Build the application for production:

```bash
npm run build
```

## Type Checking

ReScript provides robust type checking during compilation. To check types:

```bash
npm run res:build
```

## Key Components

### Analysis Workflow

1. **Data Upload**: Users can upload survival analysis datasets
2. **Analysis Configuration**: Configure survival analysis parameters
3. **Model Selection**: Choose appropriate survival analysis models
4. **Visualization**: Interactive visualizations of analysis results
5. **Sharing**: Export and share analysis results

### Visualizations

The application provides several visualization types for survival analysis:

- Kaplan-Meier Survival Curves
- Cumulative Hazard Plots
- Cox Proportional Hazards Plots
- Feature Importance Charts

## Frontend Migration Strategy

The migration from the existing React/TypeScript frontend to ReScript follows these steps:

1. **Setup**: Establish the ReScript project structure and configurations
2. **Core Components**: Migrate core UI components to ReScript
3. **State Management**: Implement state management using ReScript and React context
4. **API Integration**: Create ReScript bindings for API services
5. **Feature Parity**: Ensure all existing features are implemented in ReScript
6. **Testing**: Comprehensive testing of the migrated application
7. **Deployment**: Configure CI/CD for the new frontend

## Contributing

[Include contribution guidelines here]

## License

[Include license information here]
