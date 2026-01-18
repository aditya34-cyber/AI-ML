"use client";

import React from "react";

interface PredictionCardProps {
  prediction: {
    predicted_price: number;
    formatted_price: string;
    latitude: number;
    longitude: number;
    location_description: string;
  } | null;
  isLoading: boolean;
  error: string | null;
}

export default function PredictionCard({
  prediction,
  isLoading,
  error,
}: PredictionCardProps) {
  if (error) {
    return (
      <div className="glass-card p-8 text-center">
        <div className="text-red-400 text-lg mb-2">⚠️ Error</div>
        <p className="text-gray-400">{error}</p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="glass-card p-8">
        <div className="space-y-4">
          <div className="skeleton h-8 w-3/4 mx-auto"></div>
          <div className="skeleton h-16 w-full"></div>
          <div className="skeleton h-6 w-1/2 mx-auto"></div>
        </div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="glass-card p-8 text-center">
        <div className="text-6xl mb-4">🏠</div>
        <h3 className="text-xl font-semibold text-gray-300 mb-2">
          Select a Location
        </h3>
        <p className="text-gray-500">
          Click on the map to see the predicted house price for that area
        </p>
      </div>
    );
  }

  return (
    <div className="glass-card p-8 glass-card-hover price-reveal">
      {/* Location badge */}
      <div className="inline-flex items-center gap-2 bg-indigo-500/20 px-4 py-1.5 rounded-full mb-6">
        <span className="w-2 h-2 bg-indigo-400 rounded-full animate-pulse"></span>
        <span className="text-indigo-300 text-sm font-medium">
          {prediction.location_description}
        </span>
      </div>

      {/* Price display */}
      <div className="text-center mb-6">
        <p className="text-gray-400 text-sm mb-2">Predicted Median House Value</p>
        <h2 className="text-5xl md:text-6xl font-bold gradient-text text-glow">
          {prediction.formatted_price}
        </h2>
      </div>

      {/* Coordinates */}
      <div className="flex justify-center gap-8 text-sm">
        <div className="text-center">
          <p className="text-gray-500 mb-1">Latitude</p>
          <p className="text-indigo-400 font-mono">
            {prediction.latitude.toFixed(4)}°
          </p>
        </div>
        <div className="w-px bg-gray-700"></div>
        <div className="text-center">
          <p className="text-gray-500 mb-1">Longitude</p>
          <p className="text-cyan-400 font-mono">
            {prediction.longitude.toFixed(4)}°
          </p>
        </div>
      </div>

      {/* Decorative elements */}
      <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-indigo-500/10 to-transparent rounded-full blur-2xl -z-10"></div>
      <div className="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-tr from-cyan-500/10 to-transparent rounded-full blur-2xl -z-10"></div>
    </div>
  );
}
