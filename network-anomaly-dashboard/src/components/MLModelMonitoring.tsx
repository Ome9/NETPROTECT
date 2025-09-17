'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';
import { Brain, TrendingUp, Activity, Target, Settings, Zap, AlertCircle, Database, Clock, BarChart3 } from 'lucide-react';

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
  processingTime: number;
  predictionsPerSecond: number;
}

interface ModelPerformance {
  timestamp: number;
  accuracy: number;
  latency: number;
  throughput: number;
  confidence: number;
}

interface FeatureImportance {
  feature: string;
  importance: number;
  category: 'network' | 'behavioral' | 'temporal' | 'statistical';
}

interface MLModelMonitoringProps {
  modelName?: string;
  isActive?: boolean;
  onModelToggle?: (active: boolean) => void;
  onRetrain?: () => void;
}

export const MLModelMonitoring: React.FC<MLModelMonitoringProps> = ({
  modelName = "Anomaly Detection Neural Network",
  isActive = true,
  onModelToggle,
  onRetrain
}) => {
  const [currentMetrics, setCurrentMetrics] = useState<ModelMetrics>({
    accuracy: 0.94,
    precision: 0.91,
    recall: 0.89,
    f1Score: 0.90,
    falsePositiveRate: 0.08,
    falseNegativeRate: 0.11,
    processingTime: 12.5,
    predictionsPerSecond: 847
  });

  const [performanceHistory, setPerformanceHistory] = useState<ModelPerformance[]>([]);
  const [featureImportance] = useState<FeatureImportance[]>([
    { feature: 'Packet Rate', importance: 0.23, category: 'network' },
    { feature: 'Connection Duration', importance: 0.19, category: 'temporal' },
    { feature: 'Bytes Transferred', importance: 0.17, category: 'network' },
    { feature: 'Port Usage Pattern', importance: 0.15, category: 'behavioral' },
    { feature: 'Protocol Distribution', importance: 0.12, category: 'network' },
    { feature: 'Time of Day', importance: 0.08, category: 'temporal' },
    { feature: 'Source Entropy', importance: 0.06, category: 'statistical' },
  ]);

  const [modelStatus] = useState({
    version: "v2.1.3",
    lastTrained: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
    trainingSamples: 1247832,
    validationScore: 0.94,
    driftDetected: false,
    nextRetraining: new Date(Date.now() + 5 * 24 * 60 * 60 * 1000)
  });

  // Generate mock performance history
  useEffect(() => {
    const generateHistory = () => {
      const history: ModelPerformance[] = [];
      const now = Date.now();
      
      for (let i = 23; i >= 0; i--) {
        history.push({
          timestamp: now - i * 60 * 60 * 1000,
          accuracy: 0.92 + Math.random() * 0.06,
          latency: 10 + Math.random() * 10,
          throughput: 800 + Math.random() * 100,
          confidence: 0.85 + Math.random() * 0.10
        });
      }
      
      setPerformanceHistory(history);
    };

    generateHistory();

    // Update metrics periodically
    const interval = setInterval(() => {
      setCurrentMetrics(prev => ({
        ...prev,
        processingTime: prev.processingTime + (Math.random() - 0.5) * 2,
        predictionsPerSecond: prev.predictionsPerSecond + Math.floor((Math.random() - 0.5) * 50)
      }));

      // Add new performance data
      setPerformanceHistory(prev => {
        const newPoint: ModelPerformance = {
          timestamp: Date.now(),
          accuracy: 0.92 + Math.random() * 0.06,
          latency: 10 + Math.random() * 10,
          throughput: 800 + Math.random() * 100,
          confidence: 0.85 + Math.random() * 0.10
        };
        
        return [...prev.slice(-23), newPoint];
      });
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getMetricColor = (value: number, threshold: number, inverse = false) => {
    const isGood = inverse ? value < threshold : value > threshold;
    return isGood ? 'text-green-400' : 'text-yellow-400';
  };

  const getCategoryColor = (category: FeatureImportance['category']) => {
    switch (category) {
      case 'network': return 'bg-blue-500';
      case 'behavioral': return 'bg-purple-500';
      case 'temporal': return 'bg-green-500';
      case 'statistical': return 'bg-orange-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <motion.div 
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Enhanced Model Status Overview */}
      <Card variant="glass" className="shadow-2xl">
        <CardHeader>
          <div className="flex justify-between items-center">
            <CardTitle className="text-white flex items-center gap-2">
              <motion.div
                animate={{ rotate: isActive ? 360 : 0 }}
                transition={{ duration: 2, repeat: isActive ? Infinity : 0, ease: "linear" }}
              >
                <Brain className="h-5 w-5 text-purple-400" />
              </motion.div>
              {modelName}
              <Badge variant={isActive ? 'neon' : 'glass'} className="text-xs">
                {isActive ? 'ACTIVE' : 'INACTIVE'}
              </Badge>
            </CardTitle>
            <div className="flex gap-2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button 
                    variant="glass" 
                    size="sm"
                    onClick={onRetrain}
                    className="text-xs"
                  >
                    <Settings className="h-3 w-3 mr-1" />
                    Retrain
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Retrain model with latest data</p>
                </TooltipContent>
              </Tooltip>
              <Button 
                variant={isActive ? 'neon' : 'glass'}
                size="sm"
                onClick={() => onModelToggle?.(!isActive)}
                className="text-xs"
              >
                {isActive ? 'Stop' : 'Start'}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4 text-sm">
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="p-3 glass-effect rounded-lg"
            >
              <span className="text-gray-300">Version:</span>
              <div className="font-semibold text-blue-400 text-lg">{modelStatus.version}</div>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="p-3 glass-effect rounded-lg"
            >
              <span className="text-gray-300">Last Trained:</span>
              <div className="font-semibold text-green-400">{modelStatus.lastTrained.toLocaleDateString()}</div>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="p-3 glass-effect rounded-lg"
            >
              <span className="text-gray-300">Training Samples:</span>
              <div className="font-semibold text-purple-400">{modelStatus.trainingSamples.toLocaleString()}</div>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="p-3 glass-effect rounded-lg"
            >
              <span className="text-gray-300">Validation Score:</span>
              <div className={`font-semibold text-lg ${getMetricColor(modelStatus.validationScore, 0.9)}`}>
                {(modelStatus.validationScore * 100).toFixed(1)}%
              </div>
              <Progress 
                value={modelStatus.validationScore * 100} 
                variant="neon"
                glowColor="purple" 
                className="mt-1 h-1" 
              />
            </motion.div>
          </div>

          {modelStatus.driftDetected && (
            <div className="mt-3 p-2 bg-orange-900/20 border border-orange-500 rounded-lg">
              <div className="flex items-center gap-2 text-orange-400 text-sm">
                <AlertCircle className="h-4 w-4" />
                Data drift detected - Consider retraining the model
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Tabs defaultValue="metrics" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="metrics">Performance Metrics</TabsTrigger>
          <TabsTrigger value="charts">Performance Charts</TabsTrigger>
          <TabsTrigger value="features">Feature Importance</TabsTrigger>
        </TabsList>

        <TabsContent value="metrics" className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            {/* Enhanced Current Performance Metrics */}
            <Card variant="glass" className="neon-glow">
              <CardHeader>
                <CardTitle className="text-white text-sm flex items-center gap-2">
                  <Target className="h-4 w-4 text-green-400" />
                  Current Performance Metrics
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <motion.div 
                  className="space-y-2"
                  whileHover={{ scale: 1.02 }}
                >
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-300">Accuracy</span>
                    <span className={`font-semibold ${getMetricColor(currentMetrics.accuracy, 0.9)}`}>
                      {(currentMetrics.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={currentMetrics.accuracy * 100} variant="neon" glowColor="green" className="h-2" />
                </motion.div>

                <motion.div 
                  className="space-y-2"
                  whileHover={{ scale: 1.02 }}
                >
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-300">Precision</span>
                    <span className={`font-semibold ${getMetricColor(currentMetrics.precision, 0.85)}`}>
                      {(currentMetrics.precision * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={currentMetrics.precision * 100} variant="neon" glowColor="blue" className="h-2" />
                </motion.div>

                <motion.div 
                  className="space-y-2"
                  whileHover={{ scale: 1.02 }}
                >
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-300">Recall</span>
                    <span className={`font-semibold ${getMetricColor(currentMetrics.recall, 0.85)}`}>
                      {(currentMetrics.recall * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={currentMetrics.recall * 100} variant="neon" glowColor="green" className="h-2" />
                </motion.div>

                <motion.div 
                  className="space-y-2"
                  whileHover={{ scale: 1.02 }}
                >
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-300">F1 Score</span>
                    <span className={`font-semibold ${getMetricColor(currentMetrics.f1Score, 0.85)}`}>
                      {(currentMetrics.f1Score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={currentMetrics.f1Score * 100} variant="neon" glowColor="purple" className="h-2" />
                </motion.div>

                <Separator className="bg-gray-600" />

                <div className="grid grid-cols-2 gap-4 text-xs">
                  <div className="text-center p-2 glass-effect rounded">
                    <span className="text-gray-300">False Positive Rate</span>
                    <div className={`font-semibold text-lg ${getMetricColor(currentMetrics.falsePositiveRate, 0.1, true)}`}>
                      {(currentMetrics.falsePositiveRate * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="text-center p-2 glass-effect rounded">
                    <span className="text-gray-300">False Negative Rate</span>
                    <div className={`font-semibold text-lg ${getMetricColor(currentMetrics.falseNegativeRate, 0.1, true)}`}>
                      {(currentMetrics.falseNegativeRate * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Enhanced Real-time Performance */}
            <Card variant="glass" className="neon-glow-blue">
              <CardHeader>
                <CardTitle className="text-white text-sm flex items-center gap-2">
                  <Activity className="h-4 w-4 text-blue-400" />
                  Real-time Performance
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <motion.div 
                    className="text-center p-3 glass-effect rounded-lg"
                    whileHover={{ scale: 1.05 }}
                  >
                    <div className="flex items-center justify-center gap-1 text-xs text-gray-300 mb-1">
                      <Clock className="h-3 w-3" />
                      Processing Time
                    </div>
                    <div className={`text-2xl font-bold ${getMetricColor(20 - currentMetrics.processingTime, 15)}`}>
                      {currentMetrics.processingTime.toFixed(1)}ms
                    </div>
                  </motion.div>
                  <motion.div 
                    className="text-center p-3 glass-effect rounded-lg"
                    whileHover={{ scale: 1.05 }}
                  >
                    <div className="flex items-center justify-center gap-1 text-xs text-gray-300 mb-1">
                      <BarChart3 className="h-3 w-3" />
                      Predictions/sec
                    </div>
                    <div className={`text-2xl font-bold ${getMetricColor(currentMetrics.predictionsPerSecond, 800)}`}>
                      {currentMetrics.predictionsPerSecond}
                    </div>
                  </motion.div>
                </div>

                {performanceHistory.length > 0 && (
                  <div className="h-32 glass-effect rounded-lg p-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={performanceHistory.slice(-12)}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis 
                          dataKey="timestamp" 
                          tickFormatter={(time) => new Date(time).toLocaleTimeString().slice(0,5)}
                          stroke="#9CA3AF"
                          fontSize={10}
                        />
                        <YAxis stroke="#9CA3AF" fontSize={10} />
                        <RechartsTooltip 
                          contentStyle={{ 
                            backgroundColor: '#1F2937', 
                            border: '1px solid #374151',
                            borderRadius: '6px',
                            fontSize: '12px'
                          }}
                          labelFormatter={(time) => new Date(time as number).toLocaleTimeString()}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="latency" 
                          stroke="#3B82F6" 
                          strokeWidth={2}
                          dot={false}
                          name="Latency (ms)"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                <div className="flex justify-between text-xs text-gray-300">
                  <div className="flex items-center gap-1">
                    <Zap className="h-3 w-3 text-yellow-400" />
                    <span>Live Monitoring</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <TrendingUp className="h-3 w-3 text-green-400" />
                    <span>Last 1 hour</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="charts" className="space-y-4">
          <Card variant="glass">
            <CardHeader>
              <CardTitle className="text-white">Performance Charts</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64 glass-effect rounded-lg p-4">
                {performanceHistory.length > 0 && (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={performanceHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis 
                        dataKey="timestamp" 
                        tickFormatter={(time) => new Date(time).toLocaleTimeString().slice(0,5)}
                        stroke="#9CA3AF"
                      />
                      <YAxis stroke="#9CA3AF" />
                      <RechartsTooltip 
                        contentStyle={{ 
                          backgroundColor: '#1F2937', 
                          border: '1px solid #374151',
                          borderRadius: '6px'
                        }}
                        labelFormatter={(time) => new Date(time as number).toLocaleTimeString()}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="accuracy" 
                        stroke="#10B981" 
                        strokeWidth={2}
                        name="Accuracy"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="confidence" 
                        stroke="#3B82F6" 
                        strokeWidth={2}
                        name="Confidence"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="features" className="space-y-4">
          <Card variant="glass" className="neon-glow-purple">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Database className="h-5 w-5" />
                Feature Importance
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {featureImportance.map((feature, index) => (
                  <motion.div
                    key={feature.feature}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                    className="flex items-center gap-3"
                  >
                    <div className={`w-3 h-3 rounded-full ${getCategoryColor(feature.category)}`} />
                    <div className="flex-1">
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-300">{feature.feature}</span>
                        <span className="text-blue-400 font-semibold">
                          {(feature.importance * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress 
                        value={feature.importance * 100} 
                        variant="neon"
                        glowColor="blue" 
                        className="h-2" 
                      />
                    </div>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </motion.div>
  );
};