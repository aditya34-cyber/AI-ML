"use client";

import { useState, useCallback } from "react";
import CaliforniaMap from "@/components/CaliforniaMap";
import PredictionCard from "@/components/PredictionCard";
import InputPanel from "@/components/InputPanel";

interface Prediction {
  predicted_price: number;
  formatted_price: string;
  latitude: number;
  longitude: number;
  location_description: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function Home() {
  const [selectedLocation, setSelectedLocation] = useState<{
    lat: number;
    lng: number;
  } | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState({
    medianIncome: 3.87,
    housingAge: 28,
  });

  const fetchPrediction = useCallback(
    async (lat: number, lng: number) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch(`${API_URL}/predict`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            latitude: lat,
            longitude: lng,
            median_income: settings.medianIncome,
            housing_median_age: settings.housingAge,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || "Failed to fetch prediction");
        }

        const data = await response.json();
        setPrediction(data);
      } catch (err) {
        console.error("Prediction error:", err);
        setError(
          err instanceof Error
            ? err.message
            : "Failed to connect to API. Make sure the backend is running."
        );
      } finally {
        setIsLoading(false);
      }
    },
    [settings]
  );

  const handleLocationSelect = (lat: number, lng: number) => {
    setSelectedLocation({ lat, lng });
    fetchPrediction(lat, lng);
  };

  return (
    <div className="min-h-screen bg-gradient-animated">
      {/* Floating particles background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="particle w-64 h-64 top-20 left-10 opacity-30"></div>
        <div className="particle w-48 h-48 top-40 right-20 opacity-20" style={{ animationDelay: "5s" }}></div>
        <div className="particle w-32 h-32 bottom-40 left-1/3 opacity-25" style={{ animationDelay: "10s" }}></div>
      </div>

      {/* Main content */}
      <div className="relative z-10 container mx-auto px-4 py-8 md:py-12">
        {/* Header */}
        <header className="text-center mb-12">
          <div className="inline-flex items-center gap-2 bg-indigo-500/10 border border-indigo-500/20 rounded-full px-4 py-2 mb-6">
            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
            <span className="text-sm text-indigo-300">ML-Powered Predictions</span>
          </div>
          
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-4">
            <span className="gradient-text">California Housing</span>
            <br />
            <span className="text-gray-200">Price Predictor</span>
          </h1>
          
          <p className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto">
            Click anywhere on the map to discover predicted median house values
            powered by machine learning
          </p>
        </header>

        {/* Main grid layout */}
        <div className="grid lg:grid-cols-2 gap-8 lg:gap-12 items-start max-w-6xl mx-auto">
          {/* Left: Map */}
          <div className="order-2 lg:order-1">
            <CaliforniaMap
              onLocationSelect={handleLocationSelect}
              selectedLocation={selectedLocation}
            />
          </div>

          {/* Right: Prediction & Settings */}
          <div className="order-1 lg:order-2 space-y-6">
            <PredictionCard
              prediction={prediction}
              isLoading={isLoading}
              error={error}
            />

            <InputPanel
              settings={settings}
              onSettingsChange={setSettings}
              isVisible={showSettings}
              onToggle={() => setShowSettings(!showSettings)}
            />

            {/* Stats/Info cards */}
            <div className="grid grid-cols-2 gap-4">
              <div className="glass-card p-4 text-center">
                <p className="text-3xl font-bold text-indigo-400">20,640</p>
                <p className="text-sm text-gray-500">Training Samples</p>
              </div>
              <div className="glass-card p-4 text-center">
                <p className="text-3xl font-bold text-cyan-400">~$49K</p>
                <p className="text-sm text-gray-500">Model RMSE</p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-16 text-gray-500 text-sm">
          <p className="mb-2">
            Built with{" "}
            <span className="text-indigo-400">Random Forest</span> •{" "}
            <span className="text-cyan-400">FastAPI</span> •{" "}
            <span className="text-pink-400">Next.js</span>
          </p>
          <p className="text-gray-600">
            Based on California Housing dataset (1990 Census)
          </p>
        </footer>
      </div>
    </div>
  );
}
