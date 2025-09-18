'use client';

import React, { useEffect, useRef, useState } from 'react';

// Dynamically import leaflet only on client-side
const L = typeof window !== 'undefined' ? require('leaflet') : null;

// Fix for default markers in Leaflet with Next.js
if (typeof window !== 'undefined' && L) {
  delete (L.Icon.Default.prototype as any)._getIconUrl;
  L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  });
}

export interface GeographicData {
  country: string;
  requests: number;
  suspicious: number;
  coordinates: [number, number]; // [latitude, longitude]
}

interface LeafletWorldMapProps {
  geographicData: GeographicData[];
  className?: string;
}

export const LeafletWorldMap: React.FC<LeafletWorldMapProps> = ({ geographicData, className = "" }) => {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const [isClient, setIsClient] = useState(false);

  // Ensure we're on client side
  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    if (!isClient || !mapRef.current || mapInstanceRef.current || !L) return;

    // Initialize the map
    const map = L.map(mapRef.current, {
      center: [20, 0], // Center of the world
      zoom: 2,
      minZoom: 2,
      maxZoom: 8,
      worldCopyJump: true,
      zoomControl: true,
      scrollWheelZoom: true,
      doubleClickZoom: true,
      attributionControl: true,
    });

    mapInstanceRef.current = map;

    // Add tile layer with dark theme for cybersecurity aesthetic
    L.tileLayer('https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a>, &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors',
      maxZoom: 20,
    }).addTo(map);

    // Add markers for each geographic location
    geographicData.forEach((location) => {
      const threatLevel = (location.suspicious / location.requests) * 100;
      
      // Determine marker color and size based on threat level
      let markerColor = '#10B981'; // Green (low risk)
      let markerSize = 12;
      
      if (threatLevel > 10) {
        markerColor = '#EF4444'; // Red (high risk)
        markerSize = 18;
      } else if (threatLevel > 5) {
        markerColor = '#F59E0B'; // Orange (medium risk)
        markerSize = 15;
      } else if (threatLevel > 1) {
        markerColor = '#F59E0B'; // Yellow (low-medium risk)
        markerSize = 13;
      }

      // Create custom marker
      const customIcon = L.divIcon({
        className: 'custom-threat-marker',
        html: `
          <div style="
            background-color: ${markerColor};
            width: ${markerSize}px;
            height: ${markerSize}px;
            border-radius: 50%;
            border: 2px solid rgba(255, 255, 255, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: ${Math.max(8, markerSize - 6)}px;
            font-weight: bold;
            color: white;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
            animation: pulse-${threatLevel > 10 ? 'high' : threatLevel > 5 ? 'medium' : 'low'} 2s infinite;
          ">
            ${threatLevel > 10 ? '⚠' : threatLevel > 5 ? '!' : '●'}
          </div>
        `,
        iconSize: [markerSize, markerSize],
        iconAnchor: [markerSize / 2, markerSize / 2],
        popupAnchor: [0, -markerSize / 2],
      });

      // Create marker with popup
      const marker = L.marker(location.coordinates, { icon: customIcon });
      
      // Create detailed popup content
      const popupContent = `
        <div style="
          background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
          color: white;
          padding: 16px;
          border-radius: 12px;
          border: 1px solid #374151;
          min-width: 250px;
          box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
        ">
          <h3 style="
            margin: 0 0 12px 0;
            font-size: 18px;
            font-weight: bold;
            color: ${markerColor};
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
          ">🌍 ${location.country}</h3>
          
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
            <div style="text-align: center; padding: 8px; background: rgba(59, 130, 246, 0.1); border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.3);">
              <div style="font-size: 20px; font-weight: bold; color: #60A5FA;">${location.requests.toLocaleString()}</div>
              <div style="font-size: 12px; color: #9CA3AF;">Total Requests</div>
            </div>
            <div style="text-align: center; padding: 8px; background: rgba(239, 68, 68, 0.1); border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);">
              <div style="font-size: 20px; font-weight: bold; color: #F87171;">${location.suspicious}</div>
              <div style="font-size: 12px; color: #9CA3AF;">Suspicious</div>
            </div>
          </div>

          <div style="
            background: rgba(${markerColor === '#EF4444' ? '239, 68, 68' : markerColor === '#F59E0B' ? '245, 158, 11' : '16, 185, 129'}, 0.1);
            border: 2px solid ${markerColor};
            border-radius: 8px;
            padding: 10px;
            text-align: center;
          ">
            <div style="font-size: 14px; color: #E5E7EB; margin-bottom: 4px;">Threat Risk Level</div>
            <div style="
              font-size: 24px;
              font-weight: bold;
              color: ${markerColor};
              text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            ">
              ${threatLevel > 10 ? '🔴 HIGH' : threatLevel > 5 ? '🟡 MEDIUM' : threatLevel > 1 ? '🟢 LOW' : '✅ MINIMAL'}
            </div>
            <div style="
              font-size: 18px;
              color: ${markerColor};
              margin-top: 4px;
            ">${threatLevel.toFixed(1)}% Risk</div>
          </div>
          
          <div style="
            margin-top: 12px;
            font-size: 11px;
            color: #6B7280;
            text-align: center;
            border-top: 1px solid #374151;
            padding-top: 8px;
          ">
            📍 ${location.coordinates[0].toFixed(2)}°, ${location.coordinates[1].toFixed(2)}° | Updated: ${new Date().toLocaleTimeString()}
          </div>
        </div>
      `;

      marker.bindPopup(popupContent, {
        closeButton: true,
        autoClose: false,
        closeOnEscapeKey: true,
        className: 'custom-popup'
      });

      marker.addTo(map);

      // Add hover effects
      marker.on('mouseover', function(this: L.Marker) {
        const element = this.getElement();
        if (element) {
          element.style.setProperty('transform', 'scale(1.2)');
          element.style.setProperty('z-index', '1000');
        }
      });

      marker.on('mouseout', function(this: L.Marker) {
        const element = this.getElement();
        if (element) {
          element.style.setProperty('transform', 'scale(1.0)');
          element.style.setProperty('z-index', 'auto');
        }
      });
    });

    // Add a legend
    const legend = new (L.Control.extend({
      options: { position: 'bottomright' },
      onAdd: function() {
        const div = L.DomUtil.create('div', 'threat-legend');
        div.innerHTML = `
          <div style="
            background: rgba(17, 24, 39, 0.95);
            color: white;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #374151;
            font-size: 12px;
            backdrop-filter: blur(10px);
          ">
            <h4 style="margin: 0 0 8px 0; font-size: 14px; color: #F3F4F6;">🛡️ Threat Levels</h4>
            <div style="display: flex; align-items: center; margin: 4px 0;">
              <div style="width: 12px; height: 12px; border-radius: 50%; background: #EF4444; margin-right: 8px;"></div>
              <span>High Risk (>10%)</span>
            </div>
            <div style="display: flex; align-items: center; margin: 4px 0;">
              <div style="width: 12px; height: 12px; border-radius: 50%; background: #F59E0B; margin-right: 8px;"></div>
              <span>Medium Risk (5-10%)</span>
            </div>
            <div style="display: flex; align-items: center; margin: 4px 0;">
              <div style="width: 12px; height: 12px; border-radius: 50%; background: #10B981; margin-right: 8px;"></div>
              <span>Low Risk (<5%)</span>
            </div>
          </div>
        `;
        return div;
      }
    }))();
    
    legend.addTo(map);

    // Cleanup function
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [isClient, geographicData]);

  // Show loading state on server-side or while client is loading
  if (!isClient || !L) {
    return (
      <div 
        className={`w-full h-96 rounded-lg border border-gray-600 bg-gray-800 flex items-center justify-center ${className}`}
      >
        <div className="text-gray-400 text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400 mx-auto mb-2"></div>
          <p>Loading Interactive World Map...</p>
        </div>
      </div>
    );
  }

  return (
    <div 
      ref={mapRef} 
      className={`w-full h-96 rounded-lg border border-gray-600 ${className}`}
      style={{ background: '#1f2937' }}
    />
  );
};

export default LeafletWorldMap;