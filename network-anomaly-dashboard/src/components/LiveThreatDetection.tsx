'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { AlertTriangle, Shield, Skull, Eye, Clock, Zap, Play, Pause, Activity } from 'lucide-react';
import { networkAPI, ThreatEvent } from '../lib/api';

interface LiveThreatDetectionProps {
  threats?: ThreatEvent[];
  onThreatAction?: (threatId: string, action: 'block' | 'allow' | 'investigate') => void;
}

export const LiveThreatDetection: React.FC<LiveThreatDetectionProps> = ({ 
  threats = [],
  onThreatAction 
}) => {
  const [liveThreats, setLiveThreats] = useState<ThreatEvent[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(true);
  const [threatStats, setThreatStats] = useState({
    total: 0,
    blocked: 0,
    high: 0,
    last24h: 0
  });

  // Get threat type icon
  const getThreatIcon = (type: ThreatEvent['type']) => {
    switch (type) {
      case 'malware': return <Skull className="h-4 w-4" />;
      case 'intrusion': return <AlertTriangle className="h-4 w-4" />;
      case 'dos': return <Zap className="h-4 w-4" />;
      case 'anomaly': return <Eye className="h-4 w-4" />;
      case 'suspicious': return <Shield className="h-4 w-4" />;
      default: return <AlertTriangle className="h-4 w-4" />;
    }
  };

  // Get severity color
  const getSeverityColor = (severity: ThreatEvent['severity']) => {
    switch (severity) {
      case 'low': return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/30';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  // Set up real-time threat detection
  useEffect(() => {
    if (!isMonitoring) return;
    
    const handleThreatDetection = (threat: ThreatEvent) => {
      console.log('🚨 Real threat detected:', threat);
      setLiveThreats(prev => [threat, ...prev.slice(0, 19)]); // Keep last 20 threats
    };

    // Listen for real-time threat events from backend ML model
    networkAPI.on('threatDetection', handleThreatDetection);

    // Cleanup
    return () => {
      networkAPI.off('threatDetection', handleThreatDetection);
    };
  }, [isMonitoring]);

  // Update threat statistics
  useEffect(() => {
    const now = new Date().getTime();
    const last24h = now - 24 * 60 * 60 * 1000;
    
    setThreatStats({
      total: liveThreats.length,
      blocked: liveThreats.filter(t => t.blocked).length,
      high: liveThreats.filter(t => ['high', 'critical'].includes(t.severity)).length,
      last24h: liveThreats.filter(t => new Date(t.timestamp).getTime() > last24h).length
    });
  }, [liveThreats]);

  const handleThreatAction = (threatId: string, action: 'block' | 'allow' | 'investigate') => {
    if (onThreatAction) {
      onThreatAction(threatId, action);
    }
    
    // Update threat status locally
    setLiveThreats(prev => prev.map(threat => 
      threat.id === threatId 
        ? { ...threat, blocked: action === 'block' }
        : threat
    ));
  };

  return (
    <div className="space-y-6">
      {/* Header Controls */}
      <Card className="border-gray-800 bg-gray-900/50 backdrop-blur-sm">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center gap-2">
              <Shield className="h-5 w-5 text-blue-400" />
              Live Threat Detection
            </CardTitle>
            <Button
              onClick={() => setIsMonitoring(!isMonitoring)}
              variant={isMonitoring ? "destructive" : "default"}
              size="sm"
              className="flex items-center gap-2"
            >
              {isMonitoring ? (
                <>
                  <Pause className="h-4 w-4" />
                  Stop Monitoring
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Start Monitoring
                </>
              )}
            </Button>
          </div>
        </CardHeader>
      </Card>

      {/* Threat Statistics */}
      <div className="grid grid-cols-4 gap-4">
        <Card className="border-gray-800 bg-gray-900/50">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Total Threats</p>
                <p className="text-2xl font-bold text-white">{threatStats.total}</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-orange-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-gray-800 bg-gray-900/50">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Blocked</p>
                <p className="text-2xl font-bold text-green-400">{threatStats.blocked}</p>
              </div>
              <Shield className="h-8 w-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-gray-800 bg-gray-900/50">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">High Risk</p>
                <p className="text-2xl font-bold text-red-400">{threatStats.high}</p>
              </div>
              <Skull className="h-8 w-8 text-red-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-gray-800 bg-gray-900/50">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Last 24h</p>
                <p className="text-2xl font-bold text-blue-400">{threatStats.last24h}</p>
              </div>
              <Clock className="h-8 w-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Live Threats Feed */}
      <Card className="border-gray-800 bg-gray-900/50">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="h-5 w-5 text-green-400" />
            Live Threat Feed
            {isMonitoring && (
              <div className="flex items-center gap-1 text-sm text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                Monitoring Active
              </div>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {liveThreats.length === 0 ? (
              <div className="text-center py-8 text-gray-400">
                <Shield className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p className="text-lg font-medium">No threats detected</p>
                <p className="text-sm">
                  {isMonitoring 
                    ? "VAE autoencoder model is monitoring network traffic for anomalies..." 
                    : "Start monitoring to begin threat detection"
                  }
                </p>
              </div>
            ) : (
              liveThreats.map((threat, index) => (
                <div
                  key={threat.id}
                  className="p-4 border border-gray-700 rounded-lg bg-gray-800/30 hover:bg-gray-800/50 transition-all duration-200"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3 flex-1">
                      <div className="mt-1">
                        {getThreatIcon(threat.type)}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                          <h4 className="text-white font-medium capitalize">
                            {threat.type} Detection
                          </h4>
                          <Badge className={getSeverityColor(threat.severity)}>
                            {threat.severity}
                          </Badge>
                          {threat.blocked && (
                            <Badge className="bg-red-500/20 text-red-400 border-red-500/30">
                              Blocked
                            </Badge>
                          )}
                        </div>
                        
                        <p className="text-gray-300 text-sm mb-2 line-clamp-2">
                          {threat.description}
                        </p>
                        
                        <div className="grid grid-cols-2 gap-4 text-xs text-gray-400">
                          <div>
                            <span className="text-gray-400">Source:</span> {threat.sourceIp}
                          </div>
                          {threat.targetIp && (
                            <div>
                              <span className="text-gray-400">Target:</span> {threat.targetIp}
                            </div>
                          )}
                          <div>
                            <span className="text-gray-400">Confidence:</span> {Math.round(threat.confidence * 100)}%
                          </div>
                          <div>
                            <span className="text-gray-400">Algorithm:</span> {threat.detection.algorithm}
                          </div>
                        </div>
                        
                        <div className="mt-2">
                          <Progress 
                            value={threat.confidence * 100} 
                            className="h-1" 
                          />
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2 ml-4">
                      <div className="text-right">
                        <div className="text-xs text-gray-400">
                          {new Date(threat.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                      
                      <div className="flex gap-1">
                        {!threat.blocked && (
                          <Button
                            size="sm"
                            variant="destructive"
                            className="h-6 px-2 text-xs"
                            onClick={() => handleThreatAction(threat.id, 'block')}
                          >
                            Block
                          </Button>
                        )}
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-6 px-2 text-xs border-gray-600"
                          onClick={() => handleThreatAction(threat.id, 'investigate')}
                        >
                          Investigate
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default LiveThreatDetection;
