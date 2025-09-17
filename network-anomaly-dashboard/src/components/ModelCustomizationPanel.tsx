'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import {
  Sliders, Database, Shield, Brain,
  Play, RefreshCw, Save, Upload, Download,
  Activity, Target, Network
} from 'lucide-react';
import { networkAPI } from '../lib/api';

interface ModelSettings {
  sensitivity: number;
  alertThreshold: number;
  batchSize: number;
  learningRate: number;
  windowSize: number;
  confidence: number;
}

interface DatasetConfig {
  name: string;
  enabled: boolean;
  path: string;
  samples: number;
  features: number;
  lastUpdated: Date;
}

export const ModelCustomizationPanel: React.FC = () => {
  const [modelSettings, setModelSettings] = useState<ModelSettings>({
    sensitivity: 75,
    alertThreshold: 85,
    batchSize: 32,
    learningRate: 0.001,
    windowSize: 100,
    confidence: 90
  });

  const [datasets, setDatasets] = useState<DatasetConfig[]>([
    {
      name: 'NSL-KDD',
      enabled: true,
      path: '/datasets/nsl-kdd',
      samples: 148517,
      features: 41,
      lastUpdated: new Date('2024-09-15')
    },
    {
      name: 'CSE-CIC-IDS2018',
      enabled: true,
      path: '/datasets/cse-cic-ids2018',
      samples: 16233002,
      features: 83,
      lastUpdated: new Date('2024-09-14')
    },
    {
      name: 'UNSW-NB15',
      enabled: false,
      path: '/datasets/unsw-nb15',
      samples: 2540047,
      features: 49,
      lastUpdated: new Date('2024-09-10')
    },
    {
      name: 'Custom Dataset',
      enabled: false,
      path: '/datasets/custom',
      samples: 0,
      features: 0,
      lastUpdated: new Date()
    }
  ]);

  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [modelStatus] = useState({
    currentModel: 'Neural Network v2.1.3',
    accuracy: 94.2,
    lastTrained: new Date('2024-09-15'),
    epochs: 50,
    trainingTime: '2h 34m'
  });

  const [networkConfig, setNetworkConfig] = useState({
    monitoringInterface: 'eth0',
    captureFilter: 'tcp or udp',
    bufferSize: 1024,
    packetLimit: 10000,
    realTimeAnalysis: true,
    logForwarding: true,
    forwardingPort: 5140
  });

  const [availableInterfaces, setAvailableInterfaces] = useState<string[]>(['eth0', 'Wi-Fi']);
  const [isLoadingInterfaces, setIsLoadingInterfaces] = useState(false);
  const [monitoringStatus, setMonitoringStatus] = useState<'idle' | 'starting' | 'active' | 'error'>('idle');

  // Handle network interface change
  const handleInterfaceChange = async (interfaceName: string) => {
    setNetworkConfig(prev => ({
      ...prev,
      monitoringInterface: interfaceName
    }));
    
    // Start monitoring on selected interface
    try {
      setMonitoringStatus('starting');
      await networkAPI.startNetworkMonitoring(interfaceName);
      setMonitoringStatus('active');
      console.log(`Started monitoring on interface: ${interfaceName}`);
    } catch (error) {
      setMonitoringStatus('error');
      console.error('Failed to start monitoring:', error);
    }
  };

  // Fetch real network interfaces on component mount
  useEffect(() => {
    const fetchNetworkInterfaces = async () => {
      setIsLoadingInterfaces(true);
      try {
        const interfaces = await networkAPI.getAvailableNetworkInterfaces();
        setAvailableInterfaces(interfaces);
        // Set the first interface as default if current one is not available
        if (!interfaces.includes(networkConfig.monitoringInterface)) {
          setNetworkConfig(prev => ({
            ...prev,
            monitoringInterface: interfaces[0] || 'eth0'
          }));
        }
      } catch (error) {
        console.error('Failed to fetch network interfaces:', error);
      } finally {
        setIsLoadingInterfaces(false);
      }
    };

    fetchNetworkInterfaces();
  }, []);

  const handleSettingChange = (key: keyof ModelSettings, value: number) => {
    setModelSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleDatasetToggle = (index: number) => {
    setDatasets(prev =>
      prev.map((dataset, i) =>
        i === index ? { ...dataset, enabled: !dataset.enabled } : dataset
      )
    );
  };

  const handleTraining = () => {
    setIsTraining(true);
    setTrainingProgress(0);

    const interval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 100) {
          setIsTraining(false);
          clearInterval(interval);
          return 100;
        }
        // More realistic training progress - slower at the end
        const increment = prev < 50 ? 3 + Math.random() * 4 : 
                         prev < 90 ? 1 + Math.random() * 2 :
                         0.5 + Math.random() * 1;
        return Math.min(100, prev + increment);
      });
    }, 500);
  };

  const exportConfig = () => {
    const config = {
      modelSettings,
      datasets: datasets.filter(d => d.enabled),
      networkConfig,
      exportDate: new Date()
    };

    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'netprotect-config.json';
    a.click();
  };

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Tabs defaultValue="model" className="w-full">
        <TabsList className="grid w-full grid-cols-4 glass-effect">
          <TabsTrigger value="model" className="flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Model Settings
          </TabsTrigger>
          <TabsTrigger value="datasets" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            Datasets
          </TabsTrigger>
          <TabsTrigger value="network" className="flex items-center gap-2">
            <Shield className="h-4 w-4" />
            Network Config
          </TabsTrigger>
          <TabsTrigger value="training" className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Training
          </TabsTrigger>
        </TabsList>

        <TabsContent value="model" className="space-y-4">
          <Card variant="glass" className="shadow-2xl">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Sliders className="h-5 w-5" />
                Model Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <motion.div whileHover={{ scale: 1.02 }} className="space-y-3">
                  <label className="text-sm font-medium text-gray-300">Detection Sensitivity</label>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Conservative</span>
                      <span className="text-white font-semibold">{modelSettings.sensitivity}%</span>
                      <span className="text-gray-400">Aggressive</span>
                    </div>
                    <Progress value={modelSettings.sensitivity} variant="neon" glowColor="blue" className="h-3" />
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={modelSettings.sensitivity}
                      onChange={(e) => handleSettingChange('sensitivity', parseInt(e.target.value))}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                </motion.div>

                <motion.div whileHover={{ scale: 1.02 }} className="space-y-3">
                  <label className="text-sm font-medium text-gray-300">Alert Threshold</label>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Low</span>
                      <span className="text-white font-semibold">{modelSettings.alertThreshold}%</span>
                      <span className="text-gray-400">High</span>
                    </div>
                    <Progress value={modelSettings.alertThreshold} variant="neon" glowColor="red" className="h-3" />
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={modelSettings.alertThreshold}
                      onChange={(e) => handleSettingChange('alertThreshold', parseInt(e.target.value))}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                </motion.div>
              </div>

              <Separator className="bg-gray-600" />

              <div className="grid grid-cols-3 gap-4">
                <motion.div whileHover={{ scale: 1.02 }} className="p-4 glass-effect rounded-lg">
                  <label className="text-xs text-gray-300 mb-2 block">Batch Size</label>
                  <Input
                    type="number"
                    value={modelSettings.batchSize}
                    onChange={(e) => handleSettingChange('batchSize', parseInt(e.target.value))}
                    className="bg-transparent border-gray-600 text-white"
                  />
                </motion.div>
                <motion.div whileHover={{ scale: 1.02 }} className="p-4 glass-effect rounded-lg">
                  <label className="text-xs text-gray-300 mb-2 block">Learning Rate</label>
                  <Input
                    type="number"
                    step="0.0001"
                    value={modelSettings.learningRate}
                    onChange={(e) => handleSettingChange('learningRate', parseFloat(e.target.value))}
                    className="bg-transparent border-gray-600 text-white"
                  />
                </motion.div>
                <motion.div whileHover={{ scale: 1.02 }} className="p-4 glass-effect rounded-lg">
                  <label className="text-xs text-gray-300 mb-2 block">Window Size</label>
                  <Input
                    type="number"
                    value={modelSettings.windowSize}
                    onChange={(e) => handleSettingChange('windowSize', parseInt(e.target.value))}
                    className="bg-transparent border-gray-600 text-white"
                  />
                </motion.div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="datasets" className="space-y-4">
          <Card variant="glass" className="neon-glow">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Database className="h-5 w-5" />
                Dataset Management
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {datasets.map((dataset, index) => (
                  <motion.div
                    key={dataset.name}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                    className={`p-4 rounded-lg border transition-all duration-300 ${dataset.enabled
                      ? 'glass-effect border-2 border-blue-400'
                      : 'border-gray-600 bg-gray-800/30'
                      }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <input
                          type="checkbox"
                          checked={dataset.enabled}
                          onChange={() => handleDatasetToggle(index)}
                          className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded"
                        />
                        <div>
                          <h4 className="font-medium text-white">{dataset.name}</h4>
                          <p className="text-xs text-gray-400">{dataset.path}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm text-gray-300">
                          {dataset.samples.toLocaleString()} samples
                        </div>
                        <div className="text-xs text-gray-400">
                          {dataset.features} features
                        </div>
                      </div>
                    </div>
                    {dataset.enabled && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        className="mt-3 pt-3 border-t border-gray-600"
                      >
                        <div className="flex justify-between text-xs text-gray-400">
                          <span>Last Updated: {dataset.lastUpdated.toLocaleDateString()}</span>
                          <Badge variant="neon" className="text-xs">ACTIVE</Badge>
                        </div>
                      </motion.div>
                    )}
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="training" className="space-y-4">
          <Card variant="glass" className="neon-glow-purple">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Target className="h-5 w-5" />
                Model Training
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h4 className="text-sm font-medium text-gray-300">Current Model Status</h4>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Model:</span>
                      <span className="text-blue-400">{modelStatus.currentModel}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Accuracy:</span>
                      <span className="text-green-400 font-semibold">{modelStatus.accuracy}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Last Trained:</span>
                      <span className="text-purple-400">{modelStatus.lastTrained.toLocaleDateString()}</span>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="text-sm font-medium text-gray-300">Training Controls</h4>
                  <div className="space-y-3">
                    <Button
                      variant={isTraining ? "neon" : "glass"}
                      onClick={handleTraining}
                      disabled={isTraining}
                      className="w-full"
                    >
                      {isTraining ? (
                        <>
                          <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                          Training...
                        </>
                      ) : (
                        <>
                          <Play className="h-4 w-4 mr-2" />
                          Start Training
                        </>
                      )}
                    </Button>

                    {isTraining && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        className="space-y-2"
                      >
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-400">Progress</span>
                          <span className="text-white">{trainingProgress.toFixed(1)}%</span>
                        </div>
                        <Progress value={trainingProgress} variant="neon" glowColor="green" className="h-2" />
                      </motion.div>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="network" className="space-y-4">
          <Card variant="glass" className="neon-glow-green">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Network Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <motion.div whileHover={{ scale: 1.02 }} className="p-4 glass-effect rounded-lg">
                  <label className="text-xs text-gray-300 mb-2 block flex items-center gap-2">
                    <Network className="w-3 h-3" />
                    Monitoring Interface (like Wireshark)
                  </label>
                  <select
                    value={networkConfig.monitoringInterface}
                    onChange={(e) => handleInterfaceChange(e.target.value)}
                    className="w-full bg-gray-800/50 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-400 focus:outline-none"
                    disabled={isLoadingInterfaces || monitoringStatus === 'starting'}
                  >
                    {availableInterfaces.map((interfaceName) => (
                      <option key={interfaceName} value={interfaceName}>
                        {interfaceName}
                      </option>
                    ))}
                  </select>
                  {isLoadingInterfaces && (
                    <div className="text-xs text-gray-400 mt-1 flex items-center gap-1">
                      <RefreshCw className="w-3 h-3 animate-spin" />
                      Loading interfaces...
                    </div>
                  )}
                  {monitoringStatus === 'starting' && (
                    <div className="text-xs text-blue-400 mt-1 flex items-center gap-1">
                      <RefreshCw className="w-3 h-3 animate-spin" />
                      Starting monitoring...
                    </div>
                  )}
                  {monitoringStatus === 'active' && (
                    <div className="text-xs text-green-400 mt-1 flex items-center gap-1">
                      <Activity className="w-3 h-3" />
                      Monitoring active
                    </div>
                  )}
                  {monitoringStatus === 'error' && (
                    <div className="text-xs text-red-400 mt-1 flex items-center gap-1">
                      <Shield className="w-3 h-3" />
                      Failed to start monitoring
                    </div>
                  )}
                </motion.div>

                <motion.div whileHover={{ scale: 1.02 }} className="p-4 glass-effect rounded-lg">
                  <label className="text-xs text-gray-300 mb-2 block">Capture Filter</label>
                  <Input
                    value={networkConfig.captureFilter}
                    onChange={(e) => setNetworkConfig(prev => ({
                      ...prev,
                      captureFilter: e.target.value
                    }))}
                    className="bg-transparent border-gray-600 text-white"
                  />
                </motion.div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <motion.div whileHover={{ scale: 1.02 }} className="p-4 glass-effect rounded-lg">
                  <label className="text-xs text-gray-300 mb-2 block">Buffer Size (MB)</label>
                  <Input
                    type="number"
                    value={networkConfig.bufferSize}
                    onChange={(e) => setNetworkConfig(prev => ({
                      ...prev,
                      bufferSize: parseInt(e.target.value)
                    }))}
                    className="bg-transparent border-gray-600 text-white"
                  />
                </motion.div>

                <motion.div whileHover={{ scale: 1.02 }} className="p-4 glass-effect rounded-lg">
                  <label className="text-xs text-gray-300 mb-2 block">Forwarding Port</label>
                  <Input
                    type="number"
                    value={networkConfig.forwardingPort}
                    onChange={(e) => setNetworkConfig(prev => ({
                      ...prev,
                      forwardingPort: parseInt(e.target.value)
                    }))}
                    className="bg-transparent border-gray-600 text-white"
                  />
                </motion.div>
              </div>

              <div className="space-y-4">
                <motion.div
                  whileHover={{ scale: 1.02 }}
                  className="flex items-center justify-between p-4 glass-effect rounded-lg"
                >
                  <span className="text-sm text-gray-300">Real-time Analysis</span>
                  <button
                    onClick={() => setNetworkConfig(prev => ({
                      ...prev,
                      realTimeAnalysis: !prev.realTimeAnalysis
                    }))}
                    className={`w-12 h-6 rounded-full transition-all duration-300 ${networkConfig.realTimeAnalysis ? 'border-2 border-blue-400 bg-blue-400/20' : 'bg-gray-600'
                      }`}
                  >
                    <div className={`w-5 h-5 bg-white rounded-full transition-transform duration-300 ${networkConfig.realTimeAnalysis ? 'translate-x-6' : 'translate-x-0'
                      }`}></div>
                  </button>
                </motion.div>

                <motion.div
                  whileHover={{ scale: 1.02 }}
                  className="flex items-center justify-between p-4 glass-effect rounded-lg"
                >
                  <span className="text-sm text-gray-300">Log Forwarding</span>
                  <button
                    onClick={() => setNetworkConfig(prev => ({
                      ...prev,
                      logForwarding: !prev.logForwarding
                    }))}
                    className={`w-12 h-6 rounded-full transition-all duration-300 ${networkConfig.logForwarding ? 'border-2 border-green-400 bg-green-400/20' : 'bg-gray-600'
                      }`}
                  >
                    <div className={`w-5 h-5 bg-white rounded-full transition-transform duration-300 ${networkConfig.logForwarding ? 'translate-x-6' : 'translate-x-0'
                      }`}></div>
                  </button>
                </motion.div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Action Buttons */}
      <motion.div
        className="flex justify-end gap-3"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="glass" size="sm">
              <Upload className="h-4 w-4 mr-2" />
              Import Config
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Import configuration from file</p>
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="neon" size="sm" onClick={exportConfig}>
              <Download className="h-4 w-4 mr-2" />
              Export Config
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Export current configuration</p>
          </TooltipContent>
        </Tooltip>

        <Button variant="neon" size="sm">
          <Save className="h-4 w-4 mr-2" />
          Save Changes
        </Button>
      </motion.div>
    </motion.div>
  );
};