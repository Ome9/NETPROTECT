import React from 'react';
import { Badge } from '@/components/ui/badge';
import { AlertTriangle, Shield, Clock } from 'lucide-react';

interface AnomalyResult {
  isAnomaly: boolean;
  confidence: number;
  score: number;
  datasetType: string;
  features: number;
  timestamp?: string;
}

interface AnomalyAlertsProps {
  anomalies: AnomalyResult[];
}

export const AnomalyAlerts: React.FC<AnomalyAlertsProps> = ({ anomalies }) => {
  const formatTime = (timestamp?: string) => {
    if (!timestamp) return 'Unknown';
    return new Date(timestamp).toLocaleTimeString();
  };

  const getSeverityColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-red-400 bg-red-900/20';
    if (confidence >= 0.5) return 'text-yellow-400 bg-yellow-900/20';
    return 'text-blue-400 bg-blue-900/20';
  };

  const getSeverityText = (confidence: number) => {
    if (confidence >= 0.8) return 'HIGH';
    if (confidence >= 0.5) return 'MEDIUM';
    return 'LOW';
  };

  if (anomalies.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-gray-400">
        <Shield className="w-12 h-12 mb-4 text-green-400" />
        <h3 className="text-lg font-semibold text-gray-300">All Clear</h3>
        <p className="text-sm">No anomalies detected in recent network traffic</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {anomalies.slice(0, 10).map((anomaly, index) => (
        <div
          key={index}
          className={`p-4 rounded-lg border ${
            anomaly.isAnomaly ? 'border-red-800 bg-red-900/10' : 'border-gray-700 bg-gray-800/30'
          }`}
        >
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-3">
              {anomaly.isAnomaly ? (
                <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0" />
              ) : (
                <Shield className="w-5 h-5 text-green-400 flex-shrink-0" />
              )}
              
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <h4 className="text-sm font-semibold text-gray-200">
                    {anomaly.isAnomaly ? 'Anomaly Detected' : 'Normal Traffic'}
                  </h4>
                  <Badge 
                    variant="outline" 
                    className={`text-xs ${getSeverityColor(anomaly.confidence)}`}
                  >
                    {getSeverityText(anomaly.confidence)}
                  </Badge>
                </div>
                
                <div className="text-xs text-gray-400 space-y-1">
                  <p>Dataset: {anomaly.datasetType.toUpperCase()}</p>
                  <p>Confidence: {(anomaly.confidence * 100).toFixed(1)}%</p>
                  <p>Score: {anomaly.score.toFixed(3)}</p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-1 text-xs text-gray-400">
              <Clock className="w-3 h-3" />
              <span>{formatTime(anomaly.timestamp)}</span>
            </div>
          </div>
          
          {anomaly.isAnomaly && (
            <div className="mt-3 p-2 bg-red-900/20 rounded border border-red-800">
              <p className="text-xs text-red-300">
                Suspicious network activity detected. Review network logs and consider implementing additional security measures.
              </p>
            </div>
          )}
        </div>
      ))}
      
      {anomalies.length > 10 && (
        <div className="text-center py-2">
          <p className="text-xs text-gray-400">
            Showing 10 of {anomalies.length} alerts
          </p>
        </div>
      )}
    </div>
  );
};