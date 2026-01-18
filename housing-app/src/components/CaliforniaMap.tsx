"use client";

import React, { useState, useRef, useEffect } from "react";

interface CaliforniaMapProps {
  onLocationSelect: (lat: number, lng: number) => void;
  selectedLocation: { lat: number; lng: number } | null;
}

// California bounds (approximate)
const CA_BOUNDS = {
  minLat: 32.5,
  maxLat: 42.0,
  minLng: -124.5,
  maxLng: -114.0,
};

export default function CaliforniaMap({
  onLocationSelect,
  selectedLocation,
}: CaliforniaMapProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredCoords, setHoveredCoords] = useState<{
    lat: number;
    lng: number;
  } | null>(null);
  const [isHovering, setIsHovering] = useState(false);

  // Convert SVG coordinates to lat/lng
  const svgToLatLng = (
    svgX: number,
    svgY: number
  ): { lat: number; lng: number } => {
    // SVG viewBox is 0 0 400 500
    const lng =
      CA_BOUNDS.minLng + (svgX / 400) * (CA_BOUNDS.maxLng - CA_BOUNDS.minLng);
    const lat =
      CA_BOUNDS.maxLat - (svgY / 500) * (CA_BOUNDS.maxLat - CA_BOUNDS.minLat);
    return { lat, lng };
  };

  // Convert lat/lng to SVG coordinates
  const latLngToSvg = (
    lat: number,
    lng: number
  ): { x: number; y: number } => {
    const x =
      ((lng - CA_BOUNDS.minLng) / (CA_BOUNDS.maxLng - CA_BOUNDS.minLng)) * 400;
    const y =
      ((CA_BOUNDS.maxLat - lat) / (CA_BOUNDS.maxLat - CA_BOUNDS.minLat)) * 500;
    return { x, y };
  };

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;

    const rect = svgRef.current.getBoundingClientRect();
    const scaleX = 400 / rect.width;
    const scaleY = 500 / rect.height;

    const svgX = (e.clientX - rect.left) * scaleX;
    const svgY = (e.clientY - rect.top) * scaleY;

    const { lat, lng } = svgToLatLng(svgX, svgY);
    onLocationSelect(lat, lng);
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;

    const rect = svgRef.current.getBoundingClientRect();
    const scaleX = 400 / rect.width;
    const scaleY = 500 / rect.height;

    const svgX = (e.clientX - rect.left) * scaleX;
    const svgY = (e.clientY - rect.top) * scaleY;

    const { lat, lng } = svgToLatLng(svgX, svgY);
    setHoveredCoords({ lat, lng });
  };

  const selectedSvg = selectedLocation
    ? latLngToSvg(selectedLocation.lat, selectedLocation.lng)
    : null;

  return (
    <div className="relative w-full max-w-[500px] mx-auto">
      {/* Glow background effect */}
      <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/20 via-transparent to-cyan-500/20 rounded-3xl blur-3xl -z-10"></div>

      {/* Coordinate display on hover */}
      {isHovering && hoveredCoords && (
        <div className="absolute top-4 left-4 glass-card px-4 py-2 text-sm z-20">
          <span className="text-indigo-400">Lat:</span>{" "}
          {hoveredCoords.lat.toFixed(4)}
          <span className="mx-2 text-gray-500">|</span>
          <span className="text-cyan-400">Lng:</span>{" "}
          {hoveredCoords.lng.toFixed(4)}
        </div>
      )}

      {/* SVG Map */}
      <svg
        ref={svgRef}
        viewBox="0 0 400 500"
        className="w-full h-auto california-map cursor-crosshair"
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseEnter={() => setIsHovering(true)}
        onMouseLeave={() => setIsHovering(false)}
      >
        {/* Gradient definitions */}
        <defs>
          <linearGradient id="mapGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#4f46e5" stopOpacity="0.3" />
            <stop offset="50%" stopColor="#6366f1" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.3" />
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <radialGradient id="markerGradient" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#f472b6" />
            <stop offset="100%" stopColor="#ec4899" />
          </radialGradient>
        </defs>

        {/* California state path (simplified) */}
        <path
          d="M 50 50 
             L 120 30 
             L 180 35 
             L 220 25 
             L 260 40 
             L 300 60 
             L 340 90 
             L 360 130 
             L 370 180 
             L 365 230 
             L 355 280 
             L 340 330 
             L 310 380 
             L 270 420 
             L 220 450 
             L 170 470 
             L 120 475 
             L 80 460 
             L 50 420 
             L 35 370 
             L 30 310 
             L 35 250 
             L 40 190 
             L 45 130 
             L 48 80 
             Z"
          fill="url(#mapGradient)"
          stroke="#6366f1"
          strokeWidth="2"
          className="map-region"
          filter="url(#glow)"
        />

        {/* Major cities markers */}
        {[
          { name: "San Francisco", x: 85, y: 185, lat: 37.77, lng: -122.42 },
          { name: "Los Angeles", x: 180, y: 390, lat: 34.05, lng: -118.24 },
          { name: "San Diego", x: 220, y: 460, lat: 32.72, lng: -117.16 },
          { name: "Sacramento", x: 130, y: 155, lat: 38.58, lng: -121.49 },
          { name: "San Jose", x: 95, y: 210, lat: 37.34, lng: -121.89 },
          { name: "Fresno", x: 145, y: 270, lat: 36.74, lng: -119.79 },
        ].map((city) => (
          <g key={city.name}>
            <circle
              cx={city.x}
              cy={city.y}
              r="6"
              fill="#6366f1"
              opacity="0.6"
              className="transition-all duration-300 hover:opacity-100"
            />
            <text
              x={city.x + 10}
              y={city.y + 4}
              fill="#a5b4fc"
              fontSize="10"
              fontWeight="500"
            >
              {city.name}
            </text>
          </g>
        ))}

        {/* Selected location marker */}
        {selectedSvg && (
          <g className="pulse-marker">
            <circle
              cx={selectedSvg.x}
              cy={selectedSvg.y}
              r="20"
              fill="url(#markerGradient)"
              opacity="0.3"
            />
            <circle
              cx={selectedSvg.x}
              cy={selectedSvg.y}
              r="10"
              fill="url(#markerGradient)"
              stroke="#fff"
              strokeWidth="2"
            />
          </g>
        )}

        {/* Ocean label */}
        <text x="20" y="300" fill="#22d3ee" fontSize="14" opacity="0.5" transform="rotate(-90, 20, 300)">
          Pacific Ocean
        </text>
      </svg>

      {/* Instructions */}
      <p className="text-center text-gray-400 text-sm mt-4">
        Click anywhere on the map to get a price prediction
      </p>
    </div>
  );
}
