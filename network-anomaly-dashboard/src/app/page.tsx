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
import { UIControlPanel } from '@/components/UIControlPanel';
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
  const [currentView, setCurrentView] = useState<'overview' | 'topology' | 'threats' | 'model' | 'traffic' | 'config' | 'controls'>('overview');
  const [selectedElement, setSelectedElement] = useState<string | null>(null);
  const [isMaximized, setIsMaximized] = useState(false);
  const [topologyNodes, setTopologyNodes] = useState<Array<{
    id: string;
    ip: string;
    type: 'router' | 'server' | 'workstation' | 'unknown';
    status: 'normal' | 'suspicious' | 'threat';
    connections: number;
    lastSeen: number;
    riskScore: number;
  }>>([]);
  const [threatData, setThreatData] = useState<Array<{
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
  }>>([]);
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

  // UI Control States
  const [uiSettings, setUiSettings] = useState({
    darkMode: true,
    compactMode: false,
    animations: true,
    transparency: 85,
    glowEffects: true,
    sidebarCollapsed: false,
    fullscreenMode: false,
    gridDensity: 'normal' as 'compact' | 'normal' | 'spacious',
    colorTheme: 'cyberpunk',
    soundEnabled: true,
    notifications: true,
    alertLevel: 'medium' as 'low' | 'medium' | 'high',
    refreshRate: 10,
    dataRetention: 24,
    realtimeUpdates: true
  });

  const [showUIControls, setShowUIControls] = useState(false);

  const [visibleSections, setVisibleSections] = useState({
    systemMetrics: true,
    networkTopology: true,
    threatDetection: true,
    trafficAnalysis: true,
    modelMonitoring: true,
    configuration: true
  });

  useEffect(() => {
    setIsConnected(true);
    
    // Function to fetch real system metrics from backend API
    const fetchRealSystemMetrics = async () => {
      try {
        const { networkAPI } = await import('../lib/api');
        const realMetrics = await networkAPI.getCurrentNetworkData();
        const networkStatus = await networkAPI.getNetworkStatus();
        const topology = await networkAPI.getNetworkTopology();
        const threats = await networkAPI.getCurrentThreats();
        
        setConnectionCount(networkStatus.activeConnections);
        setTopologyNodes(topology.nodes);
        setThreatData(threats.threats);
        setSystemMetrics({
          cpuUsage: realMetrics.cpuUsage,
          memoryUsage: realMetrics.memoryUsage,
          networkLoad: realMetrics.networkLoad,
          diskUsage: realMetrics.diskUsage,
          gpuUsage: realMetrics.gpuUsage,
          threatsBlocked: realMetrics.threatsBlocked,
          activeConnections: networkStatus.activeConnections,
          modelAccuracy: realMetrics.modelAccuracy
        });
      } catch (error) {
        console.error('Failed to fetch real system metrics:', error);
        // Fallback to safe default values (not random)
        setConnectionCount(0);
        setTopologyNodes([]);
        setThreatData([]);
        setSystemMetrics({
          cpuUsage: 0,
          memoryUsage: 0,
          networkLoad: 0,
          diskUsage: 0,
          gpuUsage: 0,
          threatsBlocked: 0,
          activeConnections: 0,
          modelAccuracy: 0
        });
      }
    };
    
    // Initial fetch
    fetchRealSystemMetrics();
    
    // Use dynamic refresh rate from UI settings
    const interval = setInterval(fetchRealSystemMetrics, uiSettings.refreshRate * 1000);
    
    return () => clearInterval(interval);
  }, [uiSettings.refreshRate]);

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

  // Generate dynamic styles based on UI settings
  const getDynamicStyles = () => {
    const baseStyles = {
      backgroundColor: uiSettings.darkMode ? 'rgba(0, 0, 0, 0.8)' : 'rgba(255, 255, 255, 0.9)',
      opacity: uiSettings.transparency / 100,
      filter: uiSettings.glowEffects ? 'drop-shadow(0 0 10px rgba(59, 130, 246, 0.3))' : 'none',
    };
    
    return baseStyles;
  };

  // Color theme definitions
  const colorThemes = {
    cyberpunk: {
      primary: '#ff00ff',
      secondary: '#00ffff', 
      accent: '#ff0080',
      background: 'linear-gradient(135deg, #1a0033, #330066, #660033)',
      cardBg: 'rgba(255, 0, 255, 0.1)',
      text: '#ffffff',
      border: '#ff00ff',
      glow: '#ff00ff'
    },
    'ocean blue': {
      primary: '#0ea5e9',
      secondary: '#06b6d4',
      accent: '#0284c7', 
      background: 'linear-gradient(135deg, #001122, #003366, #004488)',
      cardBg: 'rgba(14, 165, 233, 0.1)',
      text: '#ffffff',
      border: '#0ea5e9',
      glow: '#06b6d4'
    },
    'forest green': {
      primary: '#10b981',
      secondary: '#34d399',
      accent: '#059669',
      background: 'linear-gradient(135deg, #001100, #003300, #005500)',
      cardBg: 'rgba(16, 185, 129, 0.1)',
      text: '#ffffff', 
      border: '#10b981',
      glow: '#34d399'
    },
    'sunset orange': {
      primary: '#f97316',
      secondary: '#fb923c',
      accent: '#ea580c',
      background: 'linear-gradient(135deg, #2d1b04, #5d3608, #8d5012)',
      cardBg: 'rgba(249, 115, 22, 0.1)',
      text: '#ffffff',
      border: '#f97316', 
      glow: '#fb923c'
    },
    'purple galaxy': {
      primary: '#8b5cf6',
      secondary: '#a78bfa',
      accent: '#7c3aed',
      background: 'linear-gradient(135deg, #1a0d33, #2d1a4d, #402766)',
      cardBg: 'rgba(139, 92, 246, 0.1)',
      text: '#ffffff',
      border: '#8b5cf6',
      glow: '#a78bfa'
    },
    'arctic ice': {
      primary: '#64748b',
      secondary: '#94a3b8',
      accent: '#475569',
      background: 'linear-gradient(135deg, #0f1419, #1e293b, #334155)',
      cardBg: 'rgba(100, 116, 139, 0.1)', 
      text: '#ffffff',
      border: '#64748b',
      glow: '#94a3b8'
    },
    'golden hour': {
      primary: '#f59e0b',
      secondary: '#fbbf24',
      accent: '#d97706',
      background: 'linear-gradient(135deg, #2d1a04, #5d3408, #8d4e0c)',
      cardBg: 'rgba(245, 158, 11, 0.1)',
      text: '#ffffff',
      border: '#f59e0b',
      glow: '#fbbf24'
    },
    midnight: {
      primary: '#374151',
      secondary: '#6b7280',
      accent: '#1f2937',
      background: 'linear-gradient(135deg, #000000, #111827, #1f2937)',
      cardBg: 'rgba(55, 65, 81, 0.1)',
      text: '#ffffff',
      border: '#374151',
      glow: '#6b7280'
    }
  };

  // Apply color theme to CSS variables
  const applyColorTheme = (themeName: string) => {
    const theme = colorThemes[themeName as keyof typeof colorThemes];
    if (!theme) return;

    const root = document.documentElement;
    root.style.setProperty('--theme-primary', theme.primary);
    root.style.setProperty('--theme-secondary', theme.secondary);
    root.style.setProperty('--theme-accent', theme.accent);
    root.style.setProperty('--theme-background', theme.background);
    root.style.setProperty('--theme-card-bg', theme.cardBg);
    root.style.setProperty('--theme-text', theme.text);
    root.style.setProperty('--theme-border', theme.border);
    root.style.setProperty('--theme-glow', theme.glow);
  };

  // Apply UI settings to document
  useEffect(() => {
    // Apply theme changes
    document.documentElement.classList.toggle('dark', uiSettings.darkMode);
    document.documentElement.classList.toggle('light', !uiSettings.darkMode);
    
    // Apply color theme
    applyColorTheme(uiSettings.colorTheme);
    
    // Apply animations
    document.documentElement.style.setProperty('--animations-enabled', uiSettings.animations ? '1' : '0');
    
    // Apply transparency
    document.documentElement.style.setProperty('--bg-opacity', (uiSettings.transparency / 100).toString());
    
    // Apply glow effects
    document.documentElement.style.setProperty('--glow-enabled', uiSettings.glowEffects ? '1' : '0');
    
    // Apply grid density
    const gridSpacing = uiSettings.gridDensity === 'compact' ? '1rem' : 
                       uiSettings.gridDensity === 'spacious' ? '2rem' : '1.5rem';
    document.documentElement.style.setProperty('--grid-spacing', gridSpacing);
    
  }, [uiSettings]);

  const renderOverview = () => (
    <motion.div 
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* System Status Cards - Conditional based on systemMetrics visibility */}
      {visibleSections.systemMetrics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <motion.div
          whileHover={uiSettings.animations ? { scale: 1.05 } : {}}
          transition={{ type: "spring", stiffness: 300 }}
          onClick={() => handleElementClick('network-status')}
          className="cursor-pointer"
        >
          <Card 
            variant="glass" 
            className={`group transition-all duration-300 hover-rainbow-border ${getElementClasses('network-status')} ${
              uiSettings.glowEffects ? 'hover:neon-glow-blue' : ''
            }`}
            style={{
              backgroundColor: 'var(--theme-card-bg, rgba(255, 0, 255, 0.1))',
              borderColor: 'var(--theme-border, #ff00ff)',
              boxShadow: uiSettings.glowEffects ? `0 0 20px var(--theme-glow, #ff00ff)` : 'none'
            }}
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
            <CardContent className="p-6">
              <div className="text-2xl font-bold" style={{ color: 'var(--theme-primary, #ff00ff)' }}>
                {connectionCount}
              </div>
              <p className="text-xs" style={{ color: 'var(--theme-text, #ffffff)' }}>Active connections</p>
              <Progress 
                value={75} 
                variant="neon" 
                glowColor={uiSettings.colorTheme === 'cyberpunk' ? 'purple' : 'blue'} 
                className="mt-2 h-2"
                style={{ 
                  '--progress-color': 'var(--theme-primary, #ff00ff)' 
                } as React.CSSProperties}
              />
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          whileHover={uiSettings.animations ? { scale: 1.05 } : {}}
          transition={{ type: "spring", stiffness: 300 }}
          onClick={() => handleElementClick('threat-level')}
          className="cursor-pointer"
        >
          <Card 
            variant="glass" 
            className={`group transition-all duration-300 hover-rainbow-border ${getElementClasses('threat-level')} ${
              uiSettings.glowEffects ? 'hover:neon-glow-red' : ''
            }`}
            style={{
              backgroundColor: 'var(--theme-card-bg, rgba(255, 0, 255, 0.1))',
              borderColor: 'var(--theme-border, #ff00ff)',
              boxShadow: uiSettings.glowEffects ? `0 0 20px var(--theme-secondary, #00ffff)` : 'none'
            }}
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
            <CardContent className="p-6">
              <div className="text-2xl font-bold" style={{ color: 'var(--theme-secondary, #00ffff)' }}>
                Low
              </div>
              <p className="text-xs" style={{ color: 'var(--theme-text, #ffffff)' }}>
                {systemMetrics.threatsBlocked} blocked today
              </p>
              <Progress 
                value={25} 
                variant="neon" 
                glowColor="red" 
                className="mt-2 h-2"
                style={{ 
                  '--progress-color': 'var(--theme-secondary, #00ffff)' 
                } as React.CSSProperties}
              />
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
            <CardContent className="p-6">
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
            <CardContent className="p-6">
              <div className="text-2xl font-bold text-purple-400">Active</div>
              <p className="text-xs text-gray-400">94.2% accuracy</p>
              <Progress value={94.2} variant="neon" glowColor="purple" className="mt-2 h-2" />
            </CardContent>
          </Card>
        </motion.div>
      </div>
      )}

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
        
        <div className={`container mx-auto p-6 space-y-6 pt-4 ${uiSettings.compactMode ? 'p-4 space-y-4' : 'p-6 space-y-6'}`}
             style={{
               gap: uiSettings.gridDensity === 'compact' ? '1rem' : 
                    uiSettings.gridDensity === 'spacious' ? '2rem' : '1.5rem',
               opacity: uiSettings.transparency / 100,
               filter: uiSettings.glowEffects ? 'drop-shadow(0 0 20px var(--theme-glow, rgba(59, 130, 246, 0.2)))' : 'none',
               background: `var(--theme-background, linear-gradient(135deg, #1a0033, #330066, #660033))`
             }}>
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
            <Tabs value={currentView} onValueChange={(value) => setCurrentView(value as 'overview' | 'topology' | 'threats' | 'model' | 'traffic' | 'config' | 'controls')} className="w-full">
              <TabsList className="grid w-full grid-cols-7 glass-effect">
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
                <TabsTrigger value="controls" className="flex items-center gap-2">
                  <Eye className="h-4 w-4" />
                  UI Controls
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
              
              {currentView === 'topology' && visibleSections.networkTopology && (
                <NetworkTopologyVisualizer
                  nodes={topologyNodes}
                  onNodeClick={handleNodeClick}
                />
              )}
              
              {currentView === 'threats' && visibleSections.threatDetection && (
                <LiveThreatDetection
                  threats={threatData}
                  onThreatAction={handleThreatAction}
                />
              )}
              
              {currentView === 'model' && visibleSections.modelMonitoring && (
                <MLModelMonitoring
                  modelName="NetProtect AI Neural Network"
                  isActive={true}
                  onModelToggle={handleModelToggle}
                  onRetrain={handleModelRetrain}
                />
              )}
              
              {currentView === 'traffic' && visibleSections.trafficAnalysis && (
                <AdvancedTrafficAnalyzer
                  data={[]}
                  onFilterChange={(filters) => console.log('Filters:', filters)}
                />
              )}
              
              {currentView === 'config' && visibleSections.configuration && (
                <ModelCustomizationPanel />
              )}
              
              {currentView === 'controls' && (
                <UIControlPanel 
                  currentSettings={uiSettings}
                  visibleSections={visibleSections}
                  onSettingChange={(key, value) => {
                    console.log('📊 Main Page Setting Update:', { key, value, currentState: uiSettings });
                    setUiSettings(prev => {
                      const newState = { ...prev, [key]: value };
                      console.log('📊 New UI Settings State:', newState);
                      return newState;
                    });
                    // Handle section visibility separately
                    if (key.includes('systemMetrics') || key.includes('networkTopology') || key.includes('threatDetection') || 
                        key.includes('trafficAnalysis') || key.includes('modelMonitoring') || key.includes('configuration')) {
                      setVisibleSections(prev => ({ ...prev, [key]: value as boolean }));
                    }
                  }}
                  onThemeChange={(theme) => {
                    setUiSettings(prev => ({ ...prev, darkMode: theme === 'dark' }));
                    document.documentElement.classList.toggle('light', theme === 'light');
                  }}
                  onLayoutChange={(layout) => {
                    if (layout === 'compactMode') {
                      setUiSettings(prev => ({ ...prev, compactMode: !prev.compactMode }));
                    } else if (layout === 'sidebarCollapsed') {
                      setUiSettings(prev => ({ ...prev, sidebarCollapsed: !prev.sidebarCollapsed }));
                    } else if (layout === 'fullscreenMode') {
                      setUiSettings(prev => ({ ...prev, fullscreenMode: !prev.fullscreenMode }));
                      if (!document.fullscreenElement) {
                        document.documentElement.requestFullscreen?.();
                      } else {
                        document.exitFullscreen?.();
                      }
                    }
                  }}
                  onSectionToggle={(sectionId, enabled) => {
                    setVisibleSections(prev => ({ ...prev, [sectionId]: enabled }));
                  }}
                />
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

          {/* UI Controls Overlay */}
          <AnimatePresence>
            {false && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
                onClick={() => console.log('Overlay disabled')}
              >
                <motion.div
                  initial={{ y: 50 }}
                  animate={{ y: 0 }}
                  exit={{ y: 50 }}
                  className="max-w-4xl w-full max-h-[90vh] overflow-y-auto"
                  onClick={(e) => e.stopPropagation()}
                >
                  <UIControlPanel 
                    currentSettings={uiSettings}
                    visibleSections={visibleSections}
                    onClose={() => console.log('Overlay disabled - use UI Controls tab')}
                    onSettingChange={(key, value) => {
                      console.log('📊 Main Page Setting Update:', { key, value, currentState: uiSettings });
                      setUiSettings(prev => {
                        const newState = { ...prev, [key]: value };
                        console.log('📊 New UI Settings State:', newState);
                        return newState;
                      });
                      // Handle section visibility separately
                      if (key.includes('systemMetrics') || key.includes('networkTopology') || key.includes('threatDetection') || 
                          key.includes('trafficAnalysis') || key.includes('modelMonitoring') || key.includes('configuration')) {
                        setVisibleSections(prev => ({ ...prev, [key]: value as boolean }));
                      }
                    }}
                    onThemeChange={(theme) => {
                      setUiSettings(prev => ({ ...prev, darkMode: theme === 'dark' }));
                      document.documentElement.classList.toggle('light', theme === 'light');
                    }}
                    onLayoutChange={(layout) => {
                      if (layout === 'compactMode') {
                        setUiSettings(prev => ({ ...prev, compactMode: !prev.compactMode }));
                      } else if (layout === 'sidebarCollapsed') {
                        setUiSettings(prev => ({ ...prev, sidebarCollapsed: !prev.sidebarCollapsed }));
                      } else if (layout === 'fullscreenMode') {
                        setUiSettings(prev => ({ ...prev, fullscreenMode: !prev.fullscreenMode }));
                        if (!document.fullscreenElement) {
                          document.documentElement.requestFullscreen?.();
                        } else {
                          document.exitFullscreen?.();
                        }
                      }
                    }}
                    onSectionToggle={(sectionId, enabled) => {
                      setVisibleSections(prev => ({ ...prev, [sectionId]: enabled }));
                    }}
                  />
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </AnimatedBackground>
    </TooltipProvider>
  );
}
