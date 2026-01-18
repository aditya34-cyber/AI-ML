# 🏠 California Housing Price Predictor

An interactive machine learning web application that predicts California housing prices based on location. Click anywhere on the map to get instant price predictions powered by a Random Forest model trained on the 1990 California Census housing data.

![ML-Powered](https://img.shields.io/badge/ML-Random%20Forest-green)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688)
![Frontend](https://img.shields.io/badge/Frontend-Next.js%2016-black)
![Map](https://img.shields.io/badge/Map-Leaflet-199900)

---

## ✨ Features

- 🗺️ **Interactive Map** - Click anywhere on California to get predictions
- 🤖 **ML-Powered** - Random Forest Regressor trained on 20,640+ samples
- ⚡ **Real-time API** - FastAPI backend with ~50ms response times
- 🎨 **Modern UI** - Glassmorphism design with animated gradients
- 📍 **California Boundary** - Accurate state outline overlay
- ⚙️ **Adjustable Parameters** - Customize income and housing age inputs

---

## 🏗️ Project Structure

```
AI-ML/
├── 📁 housing-app/              # Next.js Frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx         # Main application page
│   │   │   ├── layout.tsx       # Root layout with metadata
│   │   │   └── globals.css      # Tailwind & custom styles
│   │   ├── components/
│   │   │   ├── CaliforniaMap.tsx    # Map wrapper with SSR handling
│   │   │   ├── MapComponent.tsx     # Leaflet map with markers
│   │   │   ├── PredictionCard.tsx   # Price prediction display
│   │   │   └── InputPanel.tsx       # Settings panel
│   │   └── data/
│   │       └── california-boundary.ts  # State GeoJSON boundary
│   └── package.json
│
├── 📁 datasets/                  # Training data
│   ├── housing.tgz              # Original dataset archive
│   └── housing/
│       └── housing.csv          # California housing data
│
├── 🐍 api.py                     # FastAPI backend server
├── 🐍 final.py                   # ML model training script
├── 🧠 california_housing_model.pkl  # Trained model (~145MB)
├── 📋 requirements.txt           # Python dependencies
└── 📖 README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Node.js 18+
- npm or yarn

### 1️⃣ Clone & Setup Backend

```bash
# Clone the repository
git clone <your-repo-url>
cd AI-ML

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2️⃣ Train the Model (if not exists)

```bash
python final.py
```

This will:
- Download the California housing dataset
- Split data into train/test sets (80/20)
- Train a Random Forest Regressor
- Save the model as `california_housing_model.pkl`
- Display RMSE on test set (~$49,000)

### 3️⃣ Start the API Server

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: `http://localhost:8000`

### 4️⃣ Setup & Run Frontend

```bash
cd housing-app

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at: `http://localhost:3000`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status & health check |
| GET | `/health` | Health check endpoint |
| POST | `/predict` | Get housing price prediction |
| GET | `/model-info` | Model information & features |

### Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "latitude": 37.7749,
       "longitude": -122.4194,
       "median_income": 5.0,
       "housing_median_age": 30
     }'
```

### Response

```json
{
  "predicted_price": 358542.50,
  "formatted_price": "$358,543",
  "latitude": 37.7749,
  "longitude": -122.4194,
  "location_description": "San Francisco"
}
```

---

## 🧠 Machine Learning Model

### Algorithm
**Random Forest Regressor** with default hyperparameters

### Features Used
| Feature | Description |
|---------|-------------|
| `longitude` | Geographic longitude |
| `latitude` | Geographic latitude |
| `housing_median_age` | Median age of houses in the area |
| `total_rooms` | Total rooms in the block |
| `total_bedrooms` | Total bedrooms in the block |
| `population` | Population in the block |
| `households` | Number of households |
| `median_income` | Median income (in $10,000s) |
| `ocean_proximity` | Categorical: INLAND, NEAR BAY, NEAR OCEAN, <1H OCEAN, ISLAND |

### Dataset
- **Source**: California Housing dataset (1990 Census)
- **Samples**: 20,640 districts
- **Target**: Median house value ($14,999 - $500,001)

### Pipeline
1. **Numerical Features**: SimpleImputer (median) → StandardScaler
2. **Categorical Features**: SimpleImputer (most_frequent) → OneHotEncoder
3. **Model**: RandomForestRegressor

---

## 🛠️ Tech Stack

### Backend
- **FastAPI** - High-performance async Python web framework
- **Pydantic** - Data validation using Python type hints
- **Uvicorn** - ASGI server
- **scikit-learn** - Machine learning library
- **joblib** - Model serialization

### Frontend
- **Next.js 16** - React framework with App Router
- **React 19** - UI library
- **Tailwind CSS 4** - Utility-first CSS
- **Leaflet** - Interactive maps
- **react-leaflet** - React components for Leaflet

---

## 🌐 Environment Variables

### Frontend (housing-app/.env.local)

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Training Samples | 16,512 (80%) |
| Test Samples | 4,128 (20%) |
| Test RMSE | ~$49,000 |
| Prediction Range | $14,999 - $500,001 |

---

## 🎨 UI Features

- **Glassmorphism** - Frosted glass effect cards
- **Animated Gradients** - Dynamic background animations
- **Floating Particles** - Subtle animated background elements
- **Responsive Design** - Works on mobile, tablet, and desktop
- **Dark Theme** - Easy on the eyes
- **California Boundary** - GeoJSON state outline

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- Dataset from [Aurélien Géron's ML Book](https://github.com/ageron/handson-ml2)
- Map tiles from [OpenStreetMap](https://www.openstreetmap.org/)

---

<div align="center">

**Built with ❤️ for Hackathons**

</div>
