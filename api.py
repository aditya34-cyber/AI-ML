"""
California Housing Price Prediction API
FastAPI backend for serving ML predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="California Housing Price Predictor",
    description="ML-powered API to predict California housing prices based on location and features",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = Path(__file__).parent / "california_housing_model.pkl"
model = None

def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}. Run final.py first to train the model.")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

# Request/Response models
class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=32.5, le=42.0, description="Latitude (32.5-42.0 for California)")
    longitude: float = Field(..., ge=-124.5, le=-114.0, description="Longitude (-124.5 to -114.0 for California)")
    housing_median_age: Optional[float] = Field(default=28.0, ge=1, le=52, description="Median age of houses in the area")
    total_rooms: Optional[float] = Field(default=2635.0, description="Total rooms in the block")
    total_bedrooms: Optional[float] = Field(default=537.0, description="Total bedrooms in the block")
    population: Optional[float] = Field(default=1425.0, description="Population in the block")
    households: Optional[float] = Field(default=499.0, description="Number of households in the block")
    median_income: Optional[float] = Field(default=3.87, ge=0.5, le=15.0, description="Median income in $10,000s")
    ocean_proximity: Optional[str] = Field(default="INLAND", description="Ocean proximity category")

class PredictionResponse(BaseModel):
    predicted_price: float
    formatted_price: str
    latitude: float
    longitude: float
    location_description: str

# Helper to determine ocean proximity based on coordinates
def get_ocean_proximity(lat: float, lon: float) -> str:
    # Simplified logic based on California geography
    if lon < -122.0 and lat > 36.0:  # Near SF Bay
        return "<1H OCEAN"
    elif lon < -117.5:  # Coastal
        return "NEAR OCEAN"
    elif lat > 41.0 or lat < 33.5:  # Near state borders
        return "NEAR BAY"
    else:
        return "INLAND"

def get_location_description(lat: float, lon: float) -> str:
    """Get a human-readable location description based on coordinates"""
    # Major California regions
    if lat >= 37.7 and lat <= 37.85 and lon >= -122.5 and lon <= -122.35:
        return "San Francisco"
    elif lat >= 34.0 and lat <= 34.15 and lon >= -118.5 and lon <= -118.15:
        return "Los Angeles"
    elif lat >= 32.7 and lat <= 32.75 and lon >= -117.2 and lon <= -117.1:
        return "San Diego"
    elif lat >= 37.3 and lat <= 37.45 and lon >= -122.1 and lon <= -121.8:
        return "San Jose / Silicon Valley"
    elif lat >= 38.5 and lat <= 38.6 and lon >= -121.5 and lon <= -121.4:
        return "Sacramento"
    elif lat >= 36.7 and lat <= 36.8 and lon >= -119.85 and lon <= -119.75:
        return "Fresno"
    elif lat >= 33.8 and lat <= 34.2 and lon >= -118.6 and lon <= -117.5:
        return "Greater Los Angeles Area"
    elif lat >= 37.0 and lat <= 38.0 and lon >= -122.5 and lon <= -121.5:
        return "San Francisco Bay Area"
    elif lon < -122.5:
        return "Northern California Coast"
    elif lon > -117.0:
        return "Eastern California"
    elif lat > 39.0:
        return "Northern California"
    elif lat < 34.0:
        return "Southern California"
    else:
        return "Central California"

@app.get("/")
def root():
    return {
        "message": "California Housing Price Predictor API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run final.py to train and save the model first."
        )
    
    # Determine ocean proximity if not provided or use coordinate-based guess
    ocean_prox = request.ocean_proximity or get_ocean_proximity(request.latitude, request.longitude)
    
    # Prepare input data as DataFrame (matching training format)
    input_data = pd.DataFrame([{
        "longitude": request.longitude,
        "latitude": request.latitude,
        "housing_median_age": request.housing_median_age,
        "total_rooms": request.total_rooms,
        "total_bedrooms": request.total_bedrooms,
        "population": request.population,
        "households": request.households,
        "median_income": request.median_income,
        "ocean_proximity": ocean_prox
    }])
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        
        # Ensure prediction is within reasonable bounds
        prediction = max(14999, min(prediction, 500001))
        
        return PredictionResponse(
            predicted_price=round(prediction, 2),
            formatted_price=f"${prediction:,.0f}",
            latitude=request.latitude,
            longitude=request.longitude,
            location_description=get_location_description(request.latitude, request.longitude)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
def model_info():
    if model is None:
        return {"status": "Model not loaded"}
    
    return {
        "status": "Model loaded",
        "model_type": type(model).__name__,
        "features": ["longitude", "latitude", "housing_median_age", "total_rooms", 
                     "total_bedrooms", "population", "households", "median_income", "ocean_proximity"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
