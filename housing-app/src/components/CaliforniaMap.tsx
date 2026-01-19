"use client";

import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";

interface CaliforniaMapProps {
  onLocationSelect: (lat: number, lng: number) => void;
  selectedLocation: { lat: number; lng: number } | null;
}

// Dynamically import the map to avoid SSR issues with Leaflet
const MapComponent = dynamic(() => import("./MapComponent"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[500px] rounded-3xl bg-gray-800/50 flex items-center justify-center">
      <div className="text-center">
        <div className="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-gray-400">Loading map...</p>
      </div>
    </div>
  ),
});

export default function CaliforniaMap({
  onLocationSelect,
  selectedLocation,
}: CaliforniaMapProps) {
  const [hoveredCoords, setHoveredCoords] = useState<{
    lat: number;
    lng: number;
  } | null>(null);

  return (
    <div className="relative w-full">
      {/* Glow background effect */}
      <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/20 via-transparent to-cyan-500/20 rounded-3xl blur-3xl -z-10"></div>

      {/* Coordinate display */}
      {hoveredCoords && (
        <div className="absolute top-4 left-4 glass-card px-4 py-2 text-sm z-[1000]">
          <span className="text-indigo-400">Lat:</span>{" "}
          {hoveredCoords.lat.toFixed(4)}
          <span className="mx-2 text-gray-500">|</span>
          <span className="text-cyan-400">Lng:</span>{" "}
          {hoveredCoords.lng.toFixed(4)}
        </div>
      )}

      {/* Map container */}
      <div className="rounded-3xl overflow-hidden border border-indigo-500/20 shadow-lg shadow-indigo-500/10">
        <MapComponent
          onLocationSelect={onLocationSelect}
          selectedLocation={selectedLocation}
          onHover={setHoveredCoords}
        />
      </div>

      {/* Instructions */}
      <p className="text-center text-gray-400 text-sm mt-4">
        🗺️ Click anywhere on California to get a price prediction
      </p>
    </div>
  );
}
