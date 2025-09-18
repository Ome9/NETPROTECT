'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  Shield, 
  Activity, 
  Network, 
  Brain, 
  AlertTriangle, 
  Eye, 
  Lock, 
  Globe,
  TrendingUp,
  Cpu,
  Zap,
  Target,
  BarChart3,
  Settings
} from 'lucide-react';

interface NetworkNode {
  id: string;
  ip: string;
  type: 'router' | 'server' | 'workstation' | 'unknown';
  status: 'normal' | 'suspicious' | 'threat';
  connections: number;
  lastSeen: number;
  riskScore: number;
}

interface ThreatData {
  id: string;
  type: 'malware' | 'intrusion' | 'dos' | 'anomaly' | 'suspicious';
  severity: 'low' | 'medium' | 'high' | 'critical';
  sourceIp: string;
  targetIp?: string;
  timestamp: Date;
  blocked: boolean;
  confidence: number;
  description: string;
  detection: {
    algorithm: string;
  };
}

interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  networkLoad: number;
  diskUsage: number;
  gpuUsage: number;
  threatsBlocked: number;
  activeConnections: number;
  modelAccuracy: number;
}

interface LandingPageProps {
  systemMetrics: SystemMetrics;
  topologyNodes: NetworkNode[];
  threatData: ThreatData[];
  isConnected: boolean;
  connectionCount: number;
  setCurrentView: (view: 'overview' | 'landing' | 'topology' | 'threats' | 'model' | 'traffic' | 'config' | 'controls') => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({
  systemMetrics,
  topologyNodes,
  threatData,
  isConnected,
  connectionCount,
  setCurrentView
}) => {
  const activeThreats = threatData.filter(threat => !threat.blocked).length;
  const criticalThreats = threatData.filter(threat => threat.severity === 'critical' || threat.severity === 'high').length;
  const networkHealth = topologyNodes.length > 0 ? 
    (topologyNodes.filter(node => node.status === 'normal').length / topologyNodes.length) * 100 : 100;

  return (
    <div className="space-y-8">
      {/* Welcome Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center space-y-4"
      >
        <div className="flex items-center justify-center gap-3 mb-6">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          >
            <Shield className="h-12 w-12 text-blue-400" />
          </motion.div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-500 to-cyan-400 bg-clip-text text-transparent">
            NetProtect Dashboard
          </h1>
        </div>
        <p className="text-lg text-gray-300 max-w-2xl mx-auto">
          Advanced Network Anomaly Detection & Real-time Security Monitoring
        </p>
        <div className="flex items-center justify-center gap-2">
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
          <span className="text-sm text-gray-400">
            {isConnected ? `Connected - ${connectionCount} active connections` : 'Disconnected'}
          </span>
        </div>
      </motion.div>

      {/* System Status Overview */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <h2 className="text-2xl font-semibold text-white mb-6 flex items-center gap-2">
          <Activity className="h-6 w-6 text-green-400" />
          System Status Overview
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Network Health */}
          <motion.div whileHover={{ scale: 1.02 }} transition={{ type: "spring", stiffness: 300 }}>
            <Card variant="glass" className="h-full">
              <CardHeader className="pb-2">
                <CardTitle className="text-white flex items-center gap-2 text-sm">
                  <Network className="h-4 w-4 text-green-400" />
                  Network Health
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-400 mb-2">
                  {networkHealth.toFixed(1)}%
                </div>
                <Progress value={networkHealth} variant="neon" glowColor="green" className="h-2 mb-2" />
                <p className="text-xs text-gray-400">{topologyNodes.length} nodes monitored</p>
              </CardContent>
            </Card>
          </motion.div>

          {/* Active Threats */}
          <motion.div whileHover={{ scale: 1.02 }} transition={{ type: "spring", stiffness: 300 }}>
            <Card variant="glass" className="h-full">
              <CardHeader className="pb-2">
                <CardTitle className="text-white flex items-center gap-2 text-sm">
                  <AlertTriangle className="h-4 w-4 text-red-400" />
                  Active Threats
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-red-400 mb-2">
                  {activeThreats}
                </div>
                <div className="flex items-center gap-2 mb-2">
                  <Badge variant={criticalThreats > 0 ? "pulse" : "glass"} className="text-xs">
                    {criticalThreats} Critical
                  </Badge>
                </div>
                <p className="text-xs text-gray-400">{systemMetrics.threatsBlocked} blocked today</p>
              </CardContent>
            </Card>
          </motion.div>

          {/* Model Performance */}
          <motion.div whileHover={{ scale: 1.02 }} transition={{ type: "spring", stiffness: 300 }}>
            <Card variant="glass" className="h-full">
              <CardHeader className="pb-2">
                <CardTitle className="text-white flex items-center gap-2 text-sm">
                  <Brain className="h-4 w-4 text-purple-400" />
                  ML Models
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-purple-400 mb-2">
                  {systemMetrics.modelAccuracy.toFixed(1)}%
                </div>
                <Progress value={systemMetrics.modelAccuracy} variant="neon" glowColor="purple" className="h-2 mb-2" />
                <p className="text-xs text-gray-400">Average ROC-AUC Score</p>
              </CardContent>
            </Card>
          </motion.div>

          {/* System Load */}
          <motion.div whileHover={{ scale: 1.02 }} transition={{ type: "spring", stiffness: 300 }}>
            <Card variant="glass" className="h-full">
              <CardHeader className="pb-2">
                <CardTitle className="text-white flex items-center gap-2 text-sm">
                  <Cpu className="h-4 w-4 text-blue-400" />
                  System Load
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-blue-400 mb-2">
                  {Math.max(systemMetrics.cpuUsage, systemMetrics.memoryUsage).toFixed(0)}%
                </div>
                <Progress 
                  value={Math.max(systemMetrics.cpuUsage, systemMetrics.memoryUsage)} 
                  variant="neon" 
                  glowColor="blue" 
                  className="h-2 mb-2" 
                />
                <p className="text-xs text-gray-400">CPU & Memory usage</p>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </motion.div>

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        <h2 className="text-2xl font-semibold text-white mb-6 flex items-center gap-2">
          <Zap className="h-6 w-6 text-yellow-400" />
          Quick Actions
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Network Topology */}
          <motion.div whileHover={{ scale: 1.02 }} transition={{ type: "spring", stiffness: 300 }}>
            <Card variant="glass" className="h-full">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Network className="h-5 w-5 text-cyan-400" />
                  Network Topology
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-gray-300">
                  Visualize network infrastructure and monitor device status in real-time.
                </p>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-green-500"></div>
                  <span className="text-xs text-gray-400">
                    {topologyNodes.filter(n => n.status === 'normal').length} healthy nodes
                  </span>
                </div>
                <Button 
                  variant="neon" 
                  size="sm" 
                  className="w-full"
                  onClick={() => setCurrentView('topology')}
                >
                  <Globe className="h-4 w-4 mr-2" />
                  View Topology
                </Button>
              </CardContent>
            </Card>
          </motion.div>

          {/* Threat Detection */}
          <motion.div whileHover={{ scale: 1.02 }} transition={{ type: "spring", stiffness: 300 }}>
            <Card variant="glass" className="h-full">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Eye className="h-5 w-5 text-red-400" />
                  Threat Detection
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-gray-300">
                  Monitor live threats and security incidents with AI-powered detection.
                </p>
                <div className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded-full ${activeThreats > 0 ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`}></div>
                  <span className="text-xs text-gray-400">
                    {activeThreats} active threats
                  </span>
                </div>
                <Button 
                  variant="neon" 
                  size="sm" 
                  className="w-full"
                  onClick={() => setCurrentView('threats')}
                >
                  <Target className="h-4 w-4 mr-2" />
                  View Threats
                </Button>
              </CardContent>
            </Card>
          </motion.div>

          {/* ML Models */}
          <motion.div whileHover={{ scale: 1.02 }} transition={{ type: "spring", stiffness: 300 }}>
            <Card variant="glass" className="h-full">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Brain className="h-5 w-5 text-purple-400" />
                  ML Model Monitoring
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-gray-300">
                  Monitor and configure machine learning models for anomaly detection.
                </p>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-green-500"></div>
                  <span className="text-xs text-gray-400">
                    3 models active
                  </span>
                </div>
                <Button 
                  variant="neon" 
                  size="sm" 
                  className="w-full"
                  onClick={() => setCurrentView('model')}
                >
                  <Lock className="h-4 w-4 mr-2" />
                  Manage Models
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </motion.div>

      {/* Recent Activity */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.6 }}
      >
        <h2 className="text-2xl font-semibold text-white mb-6 flex items-center gap-2">
          <BarChart3 className="h-6 w-6 text-green-400" />
          Recent Activity
        </h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Latest Threats */}
          <Card variant="glass" className="h-full">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-red-400" />
                Latest Threats
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {threatData.slice(0, 3).map((threat, index) => (
                  <motion.div 
                    key={threat.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className={`flex items-center justify-between p-3 glass-effect rounded-lg border ${
                      threat.severity === 'critical' || threat.severity === 'high' ? 'border-red-500/30' : 
                      threat.severity === 'medium' ? 'border-yellow-500/30' : 'border-green-500/30'
                    }`}
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className={`text-sm font-medium ${
                          threat.severity === 'critical' || threat.severity === 'high' ? 'text-red-300' : 
                          threat.severity === 'medium' ? 'text-yellow-300' : 'text-green-300'
                        }`}>
                          {threat.type.toUpperCase()}
                        </span>
                        <Badge 
                          variant={threat.blocked ? "glass" : "pulse"} 
                          className="text-xs"
                        >
                          {threat.blocked ? 'BLOCKED' : 'ACTIVE'}
                        </Badge>
                      </div>
                      <p className="text-xs text-gray-400 mt-1">
                        {threat.sourceIp} - {new Date(threat.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </motion.div>
                ))}
                {threatData.length === 0 && (
                  <div className="text-center text-gray-400 py-8">
                    <Shield className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>No recent threats detected</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* System Performance */}
          <Card variant="glass" className="h-full">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Activity className="h-5 w-5 text-blue-400" />
                System Performance
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">CPU Usage</span>
                  <span className="text-sm font-medium text-blue-400">{systemMetrics.cpuUsage.toFixed(1)}%</span>
                </div>
                <Progress value={systemMetrics.cpuUsage} variant="neon" glowColor="blue" className="h-2" />
                
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Memory Usage</span>
                  <span className="text-sm font-medium text-green-400">{systemMetrics.memoryUsage.toFixed(1)}%</span>
                </div>
                <Progress value={systemMetrics.memoryUsage} variant="neon" glowColor="green" className="h-2" />
                
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Network Load</span>
                  <span className="text-sm font-medium text-purple-400">{systemMetrics.networkLoad.toFixed(1)}%</span>
                </div>
                <Progress value={systemMetrics.networkLoad} variant="neon" glowColor="purple" className="h-2" />
              </div>
              <Button 
                variant="neon" 
                size="sm" 
                className="w-full mt-4"
                onClick={() => setCurrentView('config')}
              >
                <Settings className="h-4 w-4 mr-2" />
                View Details
              </Button>
            </CardContent>
          </Card>
        </div>
      </motion.div>
    </div>
  );
};