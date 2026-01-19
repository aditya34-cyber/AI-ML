"use client";

import React from "react";

interface InputPanelProps {
  settings: {
    medianIncome: number;
    housingAge: number;
  };
  onSettingsChange: (settings: { medianIncome: number; housingAge: number }) => void;
  isVisible: boolean;
  onToggle: () => void;
}

export default function InputPanel({
  settings,
  onSettingsChange,
  isVisible,
  onToggle,
}: InputPanelProps) {
  return (
    <div className="glass-card overflow-hidden">
      {/* Toggle header */}
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-4 hover:bg-white/5 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="text-2xl">⚙️</span>
          <span className="font-medium text-gray-200">Advanced Settings</span>
        </div>
        <svg
          className={`w-5 h-5 text-gray-400 transition-transform duration-300 ${
            isVisible ? "rotate-180" : ""
          }`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      {/* Collapsible content */}
      <div
        className={`transition-all duration-300 ease-in-out overflow-hidden ${
          isVisible ? "max-h-96 opacity-100" : "max-h-0 opacity-0"
        }`}
      >
        <div className="p-6 pt-2 space-y-6 border-t border-white/5">
          {/* Median Income slider */}
          <div>
            <div className="flex justify-between mb-3">
              <label className="text-sm text-gray-400">Median Income</label>
              <span className="text-sm text-indigo-400 font-mono">
                ${(settings.medianIncome * 10000).toLocaleString()}
              </span>
            </div>
            <input
              type="range"
              min="0.5"
              max="15"
              step="0.1"
              value={settings.medianIncome}
              onChange={(e) =>
                onSettingsChange({
                  ...settings,
                  medianIncome: parseFloat(e.target.value),
                })
              }
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between mt-1 text-xs text-gray-500">
              <span>$5,000</span>
              <span>$150,000</span>
            </div>
          </div>

          {/* Housing Age slider */}
          <div>
            <div className="flex justify-between mb-3">
              <label className="text-sm text-gray-400">Housing Age</label>
              <span className="text-sm text-cyan-400 font-mono">
                {settings.housingAge} years
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="52"
              step="1"
              value={settings.housingAge}
              onChange={(e) =>
                onSettingsChange({
                  ...settings,
                  housingAge: parseInt(e.target.value),
                })
              }
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between mt-1 text-xs text-gray-500">
              <span>1 year</span>
              <span>52 years</span>
            </div>
          </div>

          {/* Info note */}
          <p className="text-xs text-gray-500 bg-white/5 rounded-lg p-3">
            💡 Adjust these values to see how income and housing age affect
            predicted prices in different areas.
          </p>
        </div>
      </div>

      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: linear-gradient(135deg, #6366f1, #8b5cf6);
          cursor: pointer;
          box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
          transition: all 0.2s ease;
        }
        .slider::-webkit-slider-thumb:hover {
          transform: scale(1.1);
          box-shadow: 0 0 20px rgba(99, 102, 241, 0.7);
        }
        .slider::-moz-range-thumb {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: linear-gradient(135deg, #6366f1, #8b5cf6);
          cursor: pointer;
          border: none;
          box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
        }
      `}</style>
    </div>
  );
}
