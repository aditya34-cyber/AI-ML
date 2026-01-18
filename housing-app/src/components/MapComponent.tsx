"use client";

import React, { useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMapEvents, useMap, GeoJSON } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { californiaBoundary } from "@/data/california-boundary";

// Fix for default marker icons in Next.js
const customIcon = new L.Icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

// Custom pink/accent marker for selected location
const selectedIcon = new L.Icon({
  iconUrl: "data:image/svg+xml;base64," + btoa(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 36" width="24" height="36">
      <defs>
        <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:#f472b6;stop-opacity:1" />
          <stop offset="100%" style="stop-color:#ec4899;stop-opacity:1" />
        </linearGradient>
        <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
          <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#000" flood-opacity="0.3"/>
        </filter>
      </defs>
      <path fill="url(#grad)" filter="url(#shadow)" d="M12 0C5.4 0 0 5.4 0 12c0 9 12 24 12 24s12-15 12-24c0-6.6-5.4-12-12-12z"/>
      <circle cx="12" cy="12" r="5" fill="white"/>
    </svg>
  `),
  iconSize: [30, 45],
  iconAnchor: [15, 45],
  popupAnchor: [0, -40],
});

// California bounds
const CALIFORNIA_BOUNDS: L.LatLngBoundsExpression = [
  [32.5, -124.5], // Southwest
  [42.0, -114.0], // Northeast
];

const CALIFORNIA_CENTER: L.LatLngExpression = [37.0, -119.5];

interface MapComponentProps {
  onLocationSelect: (lat: number, lng: number) => void;
  selectedLocation: { lat: number; lng: number } | null;
  onHover: (coords: { lat: number; lng: number } | null) => void;
}

// Component to handle map events
function MapEventHandler({
  onLocationSelect,
  onHover,
}: {
  onLocationSelect: (lat: number, lng: number) => void;
  onHover: (coords: { lat: number; lng: number } | null) => void;
}) {
  useMapEvents({
    click: (e) => {
      const { lat, lng } = e.latlng;
      // Check if click is within California bounds
      if (lat >= 32.5 && lat <= 42.0 && lng >= -124.5 && lng <= -114.0) {
        onLocationSelect(lat, lng);
      }
    },
    mousemove: (e) => {
      const { lat, lng } = e.latlng;
      onHover({ lat, lng });
    },
    mouseout: () => {
      onHover(null);
    },
  });
  return null;
}

// Component to fit bounds on load
function FitBounds() {
  const map = useMap();
  
  useEffect(() => {
    map.fitBounds(CALIFORNIA_BOUNDS, { padding: [20, 20] });
  }, [map]);
  
  return null;
}

export default function MapComponent({
  onLocationSelect,
  selectedLocation,
  onHover,
}: MapComponentProps) {
  return (
    <MapContainer
      center={CALIFORNIA_CENTER}
      zoom={6}
      style={{ height: "500px", width: "100%" }}
      maxBounds={[
        [30, -128],
        [45, -110],
      ]}
      minZoom={5}
      maxZoom={12}
    >
      {/* Realistic map tiles */}
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {/* California state boundary */}
      <GeoJSON
        data={californiaBoundary}
        style={{
          color: "#6366f1",
          weight: 3,
          opacity: 0.8,
          fillColor: "#818cf8",
          fillOpacity: 0.1,
        }}
      />

      <FitBounds />
      <MapEventHandler onLocationSelect={onLocationSelect} onHover={onHover} />

      {/* Selected location marker */}
      {selectedLocation && (
        <Marker
          position={[selectedLocation.lat, selectedLocation.lng]}
          icon={selectedIcon}
        >
          <Popup>
            <div className="text-center">
              <p className="font-bold text-gray-800">Selected Location</p>
              <p className="text-sm text-gray-600">
                {selectedLocation.lat.toFixed(4)}°, {selectedLocation.lng.toFixed(4)}°
              </p>
            </div>
          </Popup>
        </Marker>
      )}

      {/* Major California cities for reference */}
      {[
        { name: "San Francisco", lat: 37.7749, lng: -122.4194 },
        { name: "Los Angeles", lat: 34.0522, lng: -118.2437 },
        { name: "San Diego", lat: 32.7157, lng: -117.1611 },
        { name: "Sacramento", lat: 38.5816, lng: -121.4944 },
        { name: "San Jose", lat: 37.3382, lng: -121.8863 },
        { name: "Fresno", lat: 36.7378, lng: -119.7871 },
      ].map((city) => (
        <Marker
          key={city.name}
          position={[city.lat, city.lng]}
          icon={customIcon}
          eventHandlers={{
            click: () => onLocationSelect(city.lat, city.lng),
          }}
        >
          <Popup>
            <div className="text-center">
              <p className="font-bold text-gray-800">{city.name}</p>
              <p className="text-xs text-gray-500">Click to predict price</p>
            </div>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
