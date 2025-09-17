'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { AlertTriangle, Shield, Skull, Eye, Clock, Zap } from 'lucide-react';

interface ThreatEvent {
  id: string;
  timestamp: number;
  type: 'malware' | 'intrusion' | 'ddos' | 'phishing' | 'anomaly';
  severity: 'low' | 'medium' | 'high' | 'critical';
  sourceIp: string;
  targetIp?: string;
  description: string;
  confidence: number;
  blocked: boolean;
  modelPrediction: {
    probability: number;
    features: string[];
    algorithm: string;
  };
}

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
    critical: 0,
    last24h: 0
  });

  // Generate mock threats for demonstration
  const generateMockThreat = (): ThreatEvent => {
    const types: ThreatEvent['type'][] = ['malware', 'intrusion', 'ddos', 'phishing', 'anomaly'];
    const severities: ThreatEvent['severity'][] = ['low', 'medium', 'high', 'critical'];
    const algorithms = ['Random Forest', 'Neural Network', 'SVM', 'XGBoost'];
    
    const type = types[Math.floor(Math.random() * types.length)];
    const severity = severities[Math.floor(Math.random() * severities.length)];
    const confidence = 0.6 + Math.random() * 0.4; // 60-100%
    
    return {
      id: Date.now().toString() + Math.random(),
      timestamp: Date.now(),
      type,
      severity,
      sourceIp: `${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
      targetIp: Math.random() > 0.5 ? `192.168.1.${Math.floor(Math.random() * 255)}` : undefined,
      description: getThreatDescription(type, severity),
      confidence,
      blocked: Math.random() > 0.3,
      modelPrediction: {
        probability: confidence,
        features: getFeatures(type),
        algorithm: algorithms[Math.floor(Math.random() * algorithms.length)]
      }
    };
  };

  const getThreatDescription = (type: ThreatEvent['type'], severity: ThreatEvent['severity']): string => {
    const descriptions = {
      malware: ['Suspicious executable detected', 'Trojan signature match', 'Virus pattern identified'],
      intrusion: ['Unauthorized access attempt', 'Brute force attack detected', 'Privilege escalation'],
      ddos: ['Traffic flood detected', 'Volumetric attack identified', 'Resource exhaustion attempt'],
      phishing: ['Fraudulent domain detected', 'Social engineering attempt', 'Credential harvesting'],
      anomaly: ['Unusual network behavior', 'Abnormal data pattern', 'Statistical deviation detected']
    };
    
    const typeDescriptions = descriptions[type];
    return typeDescriptions[Math.floor(Math.random() * typeDescriptions.length)];
  };

  const getFeatures = (type: ThreatEvent['type']): string[] => {
    const featureMap = {
      malware: ['File entropy', 'API calls', 'Network connections'],
      intrusion: ['Failed logins', 'Port scans', 'Protocol anomalies'],
      ddos: ['Request rate', 'Source diversity', 'Packet size'],
      phishing: ['Domain reputation', 'SSL certificate', 'URL structure'],
      anomaly: ['Traffic patterns', 'Timing analysis', 'Behavioral model']
    };
    
    return featureMap[type] || ['Unknown feature'];
  };

  useEffect(() => {
    if (!isMonitoring) return;
    
    const interval = setInterval(() => {
      if (Math.random() > 0.7) { // 30% chance every 2 seconds
        const newThreat = generateMockThreat();
        setLiveThreats(prev => [newThreat, ...prev.slice(0, 9)]); // Keep last 10
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [isMonitoring]);

  useEffect(() => {
    // Update stats
    const now = Date.now();
    const last24h = now - 24 * 60 * 60 * 1000;
    
    setThreatStats({
      total: liveThreats.length,
      blocked: liveThreats.filter(t => t.blocked).length,
      critical: liveThreats.filter(t => t.severity === 'critical').length,
      last24h: liveThreats.filter(t => t.timestamp > last24h).length
    });
  }, [liveThreats]);

  const getSeverityColor = (severity: ThreatEvent['severity']) => {
    switch (severity) {
      case 'critical': return 'text-red-500 bg-red-900/20 border-red-500';
      case 'high': return 'text-orange-500 bg-orange-900/20 border-orange-500';
      case 'medium': return 'text-yellow-500 bg-yellow-900/20 border-yellow-500';
      default: return 'text-blue-500 bg-blue-900/20 border-blue-500';
    }
  };

  const getSeverityIcon = (severity: ThreatEvent['severity']) => {
    switch (severity) {
      case 'critical': return <Skull className="h-4 w-4" />;
      case 'high': return <AlertTriangle className="h-4 w-4" />;
      case 'medium': return <Eye className="h-4 w-4" />;
      default: return <Shield className="h-4 w-4" />;
    }
  };

  const getTypeIcon = (type: ThreatEvent['type']) => {
    switch (type) {
      case 'ddos': return <Zap className="h-4 w-4" />;
      case 'malware': return <Skull className="h-4 w-4" />;
      default: return <AlertTriangle className="h-4 w-4" />;
    }
  };

  const handleThreatAction = (threatId: string, action: 'block' | 'allow' | 'investigate') => {
    onThreatAction?.(threatId, action);
    
    if (action === 'investigate') {
      // Show investigation modal or detailed view
      const threat = liveThreats.find(t => t.id === threatId);
      if (threat) {
        alert(`Investigating Threat ${threatId}:\n\n` +
              `Type: ${threat.type}\n` +
              `Source IP: ${threat.sourceIp}\n` +
              `Target IP: ${threat.targetIp || 'N/A'}\n` +
              `Confidence: ${(threat.confidence * 100).toFixed(1)}%\n` +
              `Algorithm: ${threat.modelPrediction.algorithm}\n` +
              `Features: ${threat.modelPrediction.features.join(', ')}\n\n` +
              `Detailed investigation would show:\n` +
              `- Network traffic analysis\n` +
              `- Historical behavior patterns\n` +
              `- Threat intelligence correlation\n` +
              `- Recommended countermeasures`);
      }
      return;
    }
    
    // Update local state for block/allow actions
    setLiveThreats(prev => 
      prev.map(threat => 
        threat.id === threatId 
          ? { ...threat, blocked: action === 'block' }
          : threat
      )
    );
  };

  return (
    <div className="space-y-4">
      {/* Enhanced Stats Overview */}
      <div className="grid grid-cols-4 gap-4">
        <Card variant="glass" className="neon-glow-red">
          <CardContent className="p-3">
            <div className="text-xs text-gray-400">Total Threats</div>
            <div className="text-lg font-bold text-red-400">{threatStats.total}</div>
            <Progress value={(threatStats.total / 50) * 100} variant="neon" glowColor="red" className="mt-2 h-1" />
          </CardContent>
        </Card>
        
        <Card variant="glass" className="neon-glow-green">
          <CardContent className="p-3">
            <div className="text-xs text-gray-400">Blocked</div>
            <div className="text-lg font-bold text-green-400">{threatStats.blocked}</div>
            <Progress value={(threatStats.blocked / threatStats.total) * 100} variant="neon" glowColor="green" className="mt-2 h-1" />
          </CardContent>
        </Card>
        
        <Card variant="glass" className="neon-glow">
          <CardContent className="p-3">
            <div className="text-xs text-gray-400">Critical</div>
            <div className="text-lg font-bold text-orange-400">{threatStats.critical}</div>
            <Progress value={(threatStats.critical / 10) * 100} variant="neon" glowColor="red" className="mt-2 h-1" />
          </CardContent>
        </Card>
        
        <Card variant="glass">
          <CardContent className="p-3">
            <div className="text-xs text-white/80">Last 24h</div>
            <div className="text-lg font-bold text-white">{threatStats.last24h}</div>
            <Progress value={(threatStats.last24h / 100) * 100} variant="neon" glowColor="red" className="mt-2 h-1" />
          </CardContent>
        </Card>
      </div>

      {/* Enhanced Live Threat Feed */}
      <Card variant="glass" className="shadow-2xl">
        <CardHeader>
          <div className="flex justify-between items-center">
            <CardTitle className="text-white flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-400" />
              Live Threat Detection
              {isMonitoring && <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse ml-2" />}
            </CardTitle>
            <Button
              variant={isMonitoring ? 'glass' : 'neon'}
              size="sm"
              onClick={() => setIsMonitoring(!isMonitoring)}
              className="text-xs"
            >
              {isMonitoring ? 'Stop' : 'Start'} Monitoring
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {liveThreats.length === 0 ? (
              <div className="text-center text-gray-400 py-8">
                <Shield className="h-12 w-12 mx-auto mb-2 opacity-50" />
                <p>No threats detected</p>
                <p className="text-xs">Monitoring is {isMonitoring ? 'active' : 'paused'}</p>
              </div>
            ) : (
              liveThreats.map((threat) => (
                <div key={threat.id} 
                     className={`p-3 rounded-lg border transition-all duration-300 ${getSeverityColor(threat.severity)}`}>
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {getSeverityIcon(threat.severity)}
                      <span className="font-medium text-sm">{threat.description}</span>
                      <Badge variant={threat.blocked ? 'destructive' : 'secondary'} className="text-xs">
                        {threat.blocked ? 'BLOCKED' : 'ALLOWED'}
                      </Badge>
                    </div>
                    <div className="flex gap-1">
                      <Button 
                        size="sm" 
                        variant="outline" 
                        className="text-xs h-6"
                        onClick={() => handleThreatAction(threat.id, 'block')}
                      >
                        Block
                      </Button>
                      <Button 
                        size="sm" 
                        variant="outline" 
                        className="text-xs h-6"
                        onClick={() => handleThreatAction(threat.id, 'investigate')}
                      >
                        Investigate
                      </Button>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 text-xs mb-2">
                    <div>
                      <span className="text-gray-400">Source:</span> {threat.sourceIp}
                    </div>
                    {threat.targetIp && (
                      <div>
                        <span className="text-gray-400">Target:</span> {threat.targetIp}
                      </div>
                    )}
                    <div>
                      <span className="text-gray-400">Type:</span> 
                      <Badge variant="outline" className="text-xs ml-1">
                        {threat.type}
                      </Badge>
                    </div>
                    <div>
                      <span className="text-gray-400">Time:</span> {new Date(threat.timestamp).toLocaleTimeString()}
                    </div>
                  </div>

                  {/* ML Model Prediction Details */}
                  <div className="mt-2 p-2 bg-gray-900/30 rounded text-xs">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-gray-400">ML Confidence:</span>
                      <Progress value={threat.confidence * 100} className="flex-1 h-1" />
                      <span className="text-gray-300">{(threat.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <div>
                        <span className="text-gray-400">Algorithm:</span> {threat.modelPrediction.algorithm}
                      </div>
                      <div>
                        <span className="text-gray-400">Features:</span> {threat.modelPrediction.features.join(', ')}
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