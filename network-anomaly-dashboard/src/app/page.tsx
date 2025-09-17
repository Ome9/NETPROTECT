'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { AnimatedBackground } from '@/components/ui/animated-background';
import { Sidebar } from '@/components/ui/sidebar';
import { Fab } from '@/components/ui/fab';
import { EnhancedSlider } from '@/components/ui/enhanced-slider';
import { DemoShowcase } from '@/components/ui/demo-showcase';
import { Navbar } from '@/components/ui/navbar';
import { AlertTriangle, Activity, Network, Shield, Brain, Globe, BarChart3, Settings, Eye, TrendingUp, Lock, Cpu, Zap } from 'lucide-react';

// Import the new advanced components
import { NetworkTopologyVisualizer } from '@/components/NetworkTopologyVisualizer';
import { LiveThreatDetection } from '@/components/LiveThreatDetection';
import { MLModelMonitoring } from '@/components/MLModelMonitoring';
import { AdvancedTrafficAnalyzer } from '@/components/AdvancedTrafficAnalyzer';
import { ModelCustomizationPanel } from '@/components/ModelCustomizationPanel';
import { EnhancedSystemMetrics } from '@/components/EnhancedSystemMetrics';

export default function Dashboard() {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionCount, setConnectionCount] = useState(0);
  const [currentView, setCurrentView] = useState<'overview' | 'topology' | 'threats' | 'model' | 'traffic' | 'config' | 'demo'>('overview');
  const [selectedElement, setSelectedElement] = useState<string | null>(null);
  const [isMaximized, setIsMaximized] = useState(false);
  const [systemMetrics, setSystemMetrics] = useState({
    cpuUsage: 0,
    memoryUsage: 0,
    networkLoad: 0,
    diskUsage: 0,
    gpuUsage: 0,
    threatsBlocked: 0,
    activeConnections: 0,
    modelAccuracy: 94.2
  });

  useEffect(() => {
    setIsConnected(true);
    
    const interval = setInterval(() => {
      setConnectionCount(Math.floor(Math.random() * 100) + 10);
      setSystemMetrics({
        cpuUsage: 15 + Math.random() * 20,
        memoryUsage: 45 + Math.random() * 30,
        networkLoad: 30 + Math.random() * 40,
        diskUsage: 60 + Math.random() * 20,
        gpuUsage: 35 + Math.random() * 25,
        threatsBlocked: Math.floor(Math.random() * 5),
        activeConnections: connectionCount,
        modelAccuracy: 92 + Math.random() * 6
      });
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);

  const handleNodeClick = (node: unknown) => {
    console.log('Selected node:', node);
  };

  const handleElementClick = (elementId: string) => {
    if (selectedElement === elementId) {
      setIsMaximized(true);
    } else {
      setSelectedElement(elementId);
      setIsMaximized(false);
    }
  };

  const handleCloseElement = () => {
    setSelectedElement(null);
    setIsMaximized(false);
  };

  const getElementClasses = (elementId: string) => {
    let classes = "";
    if (selectedElement === elementId) {
      classes += " element-selected";
      if (isMaximized) {
        classes += " element-maximized";
      }
    }
    return classes;
  };

  const handleThreatAction = (threatId: string, action: 'block' | 'allow' | 'investigate') => {
    console.log('Threat action:', threatId, action);
  };

  const handleModelToggle = (active: boolean) => {
    console.log('Model toggled:', active);
  };

  const handleModelRetrain = () => {
    console.log('Model retraining initiated');
  };

  const renderOverview = () => (
    <motion.div 
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <motion.div
          whileHover={{ scale: 1.05 }}
          transition={{ type: "spring", stiffness: 300 }}
          onClick={() => handleElementClick('network-status')}
          className="cursor-pointer"
        >
          <Card 
            variant="gemini" 
            className={`group transition-all duration-300 hover-rainbow-border ${getElementClasses('network-status')}`}
          >
            {selectedElement === 'network-status' && isMaximized && (
              <button 
                onClick={(e) => { e.stopPropagation(); handleCloseElement(); }}
                className="close-button"
              >
                ×
              </button>
            )}
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-gray-300">Network Status</CardTitle>
              <Network className="h-4 w-4 text-blue-400 group-hover:text-blue-300 transition-colors" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-400">{connectionCount}</div>
              <p className="text-xs text-gray-400">Active connections</p>
              <Progress value={75} variant="neon" glowColor="blue" className="mt-2 h-2" />
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          whileHover={{ scale: 1.05 }}
          transition={{ type: "spring", stiffness: 300 }}
          onClick={() => handleElementClick('threat-level')}
          className="cursor-pointer"
        >
          <Card 
            variant="glass" 
            className={`group hover:neon-glow-red transition-all duration-300 hover-rainbow-border ${getElementClasses('threat-level')}`}
          >
            {selectedElement === 'threat-level' && isMaximized && (
              <button 
                onClick={(e) => { e.stopPropagation(); handleCloseElement(); }}
                className="close-button"
              >
                ×
              </button>
            )}
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-gray-300">Threat Level</CardTitle>
              <AlertTriangle className="h-4 w-4 text-yellow-400 group-hover:text-yellow-300 transition-colors" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-yellow-400">Low</div>
              <p className="text-xs text-gray-400">{systemMetrics.threatsBlocked} blocked today</p>
              <Progress value={25} variant="neon" glowColor="red" className="mt-2 h-2" />
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          whileHover={{ scale: 1.05 }}
          transition={{ type: "spring", stiffness: 300 }}
          onClick={() => handleElementClick('system-load')}
          className="cursor-pointer"
        >
          <Card 
            variant="glass" 
            className={`group hover:neon-glow-green transition-all duration-300 hover-rainbow-border ${getElementClasses('system-load')}`}
          >
            {selectedElement === 'system-load' && isMaximized && (
              <button 
                onClick={(e) => { e.stopPropagation(); handleCloseElement(); }}
                className="close-button"
              >
                ×
              </button>
            )}
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-gray-300">System Load</CardTitle>
              <Cpu className="h-4 w-4 text-green-400 group-hover:text-green-300 transition-colors" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-400">{systemMetrics.cpuUsage.toFixed(1)}%</div>
              <p className="text-xs text-gray-400">CPU usage</p>
              <Progress value={systemMetrics.cpuUsage} variant="neon" glowColor="green" className="mt-2 h-2" />
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          whileHover={{ scale: 1.05 }}
          transition={{ type: "spring", stiffness: 300 }}
          onClick={() => handleElementClick('ml-model')}
          className="cursor-pointer"
        >
          <Card 
            variant="glass" 
            className={`group hover:neon-glow-purple transition-all duration-300 hover-rainbow-border ${getElementClasses('ml-model')}`}
          >
            {selectedElement === 'ml-model' && isMaximized && (
              <button 
                onClick={(e) => { e.stopPropagation(); handleCloseElement(); }}
                className="close-button"
              >
                ×
              </button>
            )}
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-gray-300">ML Model</CardTitle>
              <Brain className="h-4 w-4 text-purple-400 group-hover:text-purple-300 transition-colors" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-400">Active</div>
              <p className="text-xs text-gray-400">94.2% accuracy</p>
              <Progress value={94.2} variant="neon" glowColor="purple" className="mt-2 h-2" />
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Enhanced System Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <motion.div
          whileHover={{ scale: 1.02 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          <Card variant="glass" className="h-full">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Eye className="h-5 w-5" />
                Live Threat Feed
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <motion.div 
                  className="flex items-center justify-between p-3 glass-effect rounded-lg border border-red-500/30"
                  whileHover={{ scale: 1.02 }}
                >
                  <div>
                    <span className="text-sm text-red-300 font-medium">Suspicious activity detected</span>
                    <p className="text-xs text-gray-400">192.168.1.150 - Unusual port scanning</p>
                  </div>
                  <Badge variant="pulse" className="text-xs">CRITICAL</Badge>
                </motion.div>
                <motion.div 
                  className="flex items-center justify-between p-3 glass-effect rounded-lg border border-yellow-500/30"
                  whileHover={{ scale: 1.02 }}
                >
                  <div>
                    <span className="text-sm text-yellow-300 font-medium">Anomaly pattern identified</span>
                    <p className="text-xs text-gray-400">10.0.0.50 - High bandwidth usage</p>
                  </div>
                  <Badge variant="neon" className="text-xs">MEDIUM</Badge>
                </motion.div>
              </div>
              <Button 
                variant="glass" 
                size="sm" 
                className="w-full mt-4"
                onClick={() => setCurrentView('threats')}
              >
                <TrendingUp className="h-4 w-4 mr-2" />
                View All Threats
              </Button>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          whileHover={{ scale: 1.02 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          <Card variant="glass" className="h-full neon-glow">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Network className="h-5 w-5" />
                Network Topology
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-32 cyber-grid rounded-lg flex items-center justify-center relative overflow-hidden">
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="grid grid-cols-3 gap-4">
                    {[1, 2, 3, 4, 5, 6].map((i) => (
                      <motion.div 
                        key={i} 
                        className={`w-6 h-6 rounded-full ${
                          i === 3 ? 'bg-red-500 neon-glow-red' : 
                          i === 2 ? 'bg-yellow-500' : 'bg-green-500 neon-glow-green'
                        } flex items-center justify-center cursor-pointer`}
                        whileHover={{ scale: 1.2 }}
                        animate={{ 
                          scale: i === 3 ? [1, 1.1, 1] : 1,
                        }}
                        transition={{ 
                          duration: 2, 
                          repeat: i === 3 ? Infinity : 0 
                        }}
                      >
                        <div className="w-2 h-2 bg-white rounded-full"></div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>
              <Button 
                variant="neon" 
                size="sm" 
                className="w-full mt-4"
                onClick={() => setCurrentView('topology')}
              >
                <Globe className="h-4 w-4 mr-2" />
                View Full Topology
              </Button>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          whileHover={{ scale: 1.02 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          <Card variant="glass" className="h-full neon-glow-purple">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Brain className="h-5 w-5" />
                AI Model Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-300">NSL-KDD Model</span>
                  <Badge variant="glass" className="text-xs">95.17% ROC-AUC</Badge>
                </div>
                <Progress value={95.17} variant="neon" glowColor="green" className="h-2" />
                
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-300">CSE-CIC Model</span>
                  <Badge variant="neon" className="text-xs">90.03% ROC-AUC</Badge>
                </div>
                <Progress value={90.03} variant="neon" glowColor="blue" className="h-2" />
                
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-300">UNSW Model</span>
                  <Badge variant="glass" className="text-xs">91.84% ROC-AUC</Badge>
                </div>
                <Progress value={91.84} variant="neon" glowColor="green" className="h-2" />
              </div>
              <Button 
                variant="neon" 
                size="sm" 
                className="w-full mt-4"
                onClick={() => setCurrentView('model')}
              >
                <Lock className="h-4 w-4 mr-2" />
                Monitor Models
              </Button>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Enhanced System Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <EnhancedSystemMetrics metrics={systemMetrics} />
      </motion.div>
    </motion.div>
  );

  return (
    <TooltipProvider>
      {/* Modal Overlay for maximized elements */}
      <div className={`modal-overlay ${isMaximized ? 'active' : ''}`} onClick={handleCloseElement} />
      
      <AnimatedBackground className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
        {/* Modern Navbar */}
        <Navbar 
          currentView={currentView}
          onViewChange={(view) => setCurrentView(view as typeof currentView)}
          isConnected={isConnected}
          threatCount={systemMetrics.threatsBlocked}
          userName="Security Admin"
        />

        {/* Enhanced Sidebar */}
        <Sidebar 
          currentView={currentView}
          onViewChange={(view) => setCurrentView(view as typeof currentView)}
          isConnected={isConnected}
          threatCount={systemMetrics.threatsBlocked}
        />
        
        {/* Floating Action Button */}
        <Fab 
          icon="zap"
          variant="neon"
          tooltip="Quick Settings"
          onClick={() => setCurrentView('config')}
        />
        
        <div className="container mx-auto p-6 space-y-6 pt-4">
          {/* Enhanced Header */}
          <motion.div 
            className="flex justify-between items-start"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div>
              <motion.h1 
                className="text-4xl font-bold text-white mb-2"
                animate={{ 
                  backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"] 
                }}
                transition={{ 
                  duration: 3, 
                  repeat: Infinity, 
                  ease: "linear" 
                }}
              >
                NetProtect AI - Advanced Network Security
              </motion.h1>
              <p className="text-gray-300 text-lg">Real-time ML-powered network anomaly detection and threat analysis</p>
            </div>
            
            <motion.div 
              className="flex items-center gap-3"
              whileHover={{ scale: 1.05 }}
            >
              <Badge variant={isConnected ? "neon" : "pulse"} className="text-sm px-3 py-1">
                {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
              </Badge>
              {isConnected && (
                <motion.div 
                  className="w-3 h-3 bg-green-400 rounded-full"
                  animate={{ scale: [1, 1.2, 1], opacity: [1, 0.7, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              )}
            </motion.div>
          </motion.div>

          {/* Enhanced Navigation with Tabs */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Tabs value={currentView} onValueChange={(value) => setCurrentView(value as 'overview' | 'topology' | 'threats' | 'model' | 'traffic' | 'config')} className="w-full">
              <TabsList className="grid w-full grid-cols-6 glass-effect">
                <TabsTrigger value="overview" className="flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  Overview
                </TabsTrigger>
                <TabsTrigger value="topology" className="flex items-center gap-2">
                  <Network className="h-4 w-4" />
                  Topology
                </TabsTrigger>
                <TabsTrigger value="threats" className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4" />
                  Threats
                </TabsTrigger>
                <TabsTrigger value="model" className="flex items-center gap-2">
                  <Brain className="h-4 w-4" />
                  ML Models
                </TabsTrigger>
                <TabsTrigger value="traffic" className="flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Traffic
                </TabsTrigger>
                <TabsTrigger value="config" className="flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  Config
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </motion.div>

          {/* Dynamic Content with AnimatePresence */}
          <AnimatePresence mode="wait">
            <motion.div 
              key={currentView}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
            >
              {currentView === 'overview' && renderOverview()}
              
              {currentView === 'topology' && (
                <NetworkTopologyVisualizer
                  nodes={[]}
                  onNodeClick={handleNodeClick}
                />
              )}
              
              {currentView === 'threats' && (
                <LiveThreatDetection
                  threats={[]}
                  onThreatAction={handleThreatAction}
                />
              )}
              
              {currentView === 'model' && (
                <MLModelMonitoring
                  modelName="NetProtect AI Neural Network"
                  isActive={true}
                  onModelToggle={handleModelToggle}
                  onRetrain={handleModelRetrain}
                />
              )}
              
              {currentView === 'traffic' && (
                <AdvancedTrafficAnalyzer
                  data={[]}
                  onFilterChange={(filters) => console.log('Filters:', filters)}
                />
              )}
              
              {currentView === 'config' && (
                <ModelCustomizationPanel />
              )}
              
              {currentView === 'demo' && (
                <DemoShowcase />
              )}
            </motion.div>
          </AnimatePresence>

          {/* Enhanced Status Footer */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Card variant="glass" className="mt-8">
              <CardContent className="p-4">
                <div className="flex justify-between items-center text-white text-sm">
                  <div className="flex items-center gap-6">
                    <Tooltip>
                      <TooltipTrigger>
                        <div className="flex items-center gap-2 hover:scale-105 transition-transform">
                          <Shield className="h-4 w-4 text-green-400" />
                          <span>Security Status: Active</span>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>All security systems operational</p>
                      </TooltipContent>
                    </Tooltip>
                    
                    <Separator orientation="vertical" className="h-4 bg-white/20" />
                    
                    <Tooltip>
                      <TooltipTrigger>
                        <div className="flex items-center gap-2 hover:scale-105 transition-transform">
                          <Zap className="h-4 w-4 text-yellow-400" />
                          <span>ML Engine: Running</span>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>3 models active with 92.3% avg accuracy</p>
                      </TooltipContent>
                    </Tooltip>
                    
                    <Separator orientation="vertical" className="h-4 bg-white/20" />
                    
                    <Tooltip>
                      <TooltipTrigger>
                        <div className="flex items-center gap-2 hover:scale-105 transition-transform">
                          <Globe className="h-4 w-4 text-blue-400" />
                          <span>Backend: Connected (Port 5000)</span>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>WebSocket connection stable</p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  
                  <motion.div 
                    className="text-xs text-gray-300"
                    animate={{ opacity: [1, 0.7, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    Last updated: Live
                  </motion.div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </AnimatedBackground>
    </TooltipProvider>
  );
}
