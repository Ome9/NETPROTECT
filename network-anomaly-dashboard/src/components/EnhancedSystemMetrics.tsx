'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { 
  Cpu, 
  HardDrive, 
  Wifi, 
  Zap, 
  Activity, 
  TrendingUp, 
  TrendingDown,
  Server,
  Database,
  Network
} from 'lucide-react';

interface SystemMetricsProps {
  metrics: {
    cpuUsage: number;
    memoryUsage: number;
    networkLoad: number;
    diskUsage: number;
    gpuUsage: number;
    threatsBlocked: number;
    activeConnections: number;
    modelAccuracy: number;
  };
}

export const EnhancedSystemMetrics: React.FC<SystemMetricsProps> = ({ metrics }) => {
  const getUsageColor = (usage: number): 'red' | 'green' | 'blue' => {
    if (usage > 80) return 'red';
    if (usage > 60) return 'blue'; // Changed from yellow to blue since yellow isn't supported
    return 'green';
  };

  const getUsageVariant = (usage: number) => {
    if (usage > 80) return 'neon';
    if (usage > 60) return 'neon';
    return 'rainbow';
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {/* CPU Usage */}
      <motion.div
        whileHover={{ scale: 1.05, rotateY: 5 }}
        transition={{ type: "spring", stiffness: 300 }}
      >
        <Card variant="glass" className={`neon-glow-${getUsageColor(metrics.cpuUsage)} group`}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-300">CPU Usage</CardTitle>
            <Tooltip>
              <TooltipTrigger>
                <Cpu className={`h-4 w-4 text-${getUsageColor(metrics.cpuUsage)}-400 group-hover:animate-spin`} />
              </TooltipTrigger>
              <TooltipContent>
                <p>Current CPU utilization across all cores</p>
              </TooltipContent>
            </Tooltip>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-400">{metrics.cpuUsage.toFixed(1)}%</div>
            <Progress 
              value={metrics.cpuUsage} 
              variant={getUsageVariant(metrics.cpuUsage)} 
              glowColor={getUsageColor(metrics.cpuUsage)} 
              className="mt-2 h-2" 
            />
            <div className="flex items-center mt-2">
              {metrics.cpuUsage > 70 ? (
                <TrendingUp className="h-3 w-3 text-red-400 mr-1" />
              ) : (
                <TrendingDown className="h-3 w-3 text-green-400 mr-1" />
              )}
              <p className="text-xs text-gray-400">
                {metrics.cpuUsage > 70 ? 'High load' : 'Normal operation'}
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Memory Usage */}
      <motion.div
        whileHover={{ scale: 1.05, rotateY: 5 }}
        transition={{ type: "spring", stiffness: 300 }}
      >
        <Card variant="glass" className={`neon-glow-${getUsageColor(metrics.memoryUsage)} group`}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-300">Memory Usage</CardTitle>
            <Tooltip>
              <TooltipTrigger>
                <HardDrive className={`h-4 w-4 text-${getUsageColor(metrics.memoryUsage)}-400 group-hover:animate-pulse`} />
              </TooltipTrigger>
              <TooltipContent>
                <p>System RAM utilization</p>
              </TooltipContent>
            </Tooltip>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-400">{metrics.memoryUsage.toFixed(1)}%</div>
            <Progress 
              value={metrics.memoryUsage} 
              variant={getUsageVariant(metrics.memoryUsage)} 
              glowColor={getUsageColor(metrics.memoryUsage)} 
              className="mt-2 h-2" 
            />
            <div className="flex items-center mt-2">
              <Database className="h-3 w-3 text-blue-400 mr-1" />
              <p className="text-xs text-gray-400">
                {(metrics.memoryUsage * 16 / 100).toFixed(1)}GB / 16GB
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Network Load */}
      <motion.div
        whileHover={{ scale: 1.05, rotateY: 5 }}
        transition={{ type: "spring", stiffness: 300 }}
      >
        <Card variant="glass" className={`neon-glow-${getUsageColor(metrics.networkLoad)} group`}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-300">Network Load</CardTitle>
            <Tooltip>
              <TooltipTrigger>
                <Wifi className={`h-4 w-4 text-${getUsageColor(metrics.networkLoad)}-400 group-hover:animate-bounce`} />
              </TooltipTrigger>
              <TooltipContent>
                <p>Network bandwidth utilization</p>
              </TooltipContent>
            </Tooltip>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-400">{metrics.networkLoad.toFixed(1)}%</div>
            <Progress 
              value={metrics.networkLoad} 
              variant={getUsageVariant(metrics.networkLoad)} 
              glowColor={getUsageColor(metrics.networkLoad)} 
              className="mt-2 h-2" 
            />
            <div className="flex items-center mt-2">
              <Network className="h-3 w-3 text-purple-400 mr-1" />
              <p className="text-xs text-gray-400">
                {metrics.activeConnections} active connections
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* GPU Usage */}
      <motion.div
        whileHover={{ scale: 1.05, rotateY: 5 }}
        transition={{ type: "spring", stiffness: 300 }}
      >
        <Card variant="rainbow" className="group">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-white">GPU Usage</CardTitle>
            <Tooltip>
              <TooltipTrigger>
                <Zap className="h-4 w-4 text-yellow-400 group-hover:animate-pulse" />
              </TooltipTrigger>
              <TooltipContent>
                <p>GPU utilization for ML processing</p>
              </TooltipContent>
            </Tooltip>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{metrics.gpuUsage.toFixed(1)}%</div>
            <Progress 
              value={metrics.gpuUsage} 
              variant="rainbow" 
              className="mt-2 h-2" 
            />
            <div className="flex items-center mt-2">
              <Activity className="h-3 w-3 text-yellow-400 mr-1" />
              <p className="text-xs text-white/80">
                ML models active
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Additional Metrics Row */}
      <motion.div
        whileHover={{ scale: 1.05 }}
        transition={{ type: "spring", stiffness: 300 }}
        className="md:col-span-2"
      >
        <Card variant="glass" className="neon-glow">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Server className="h-5 w-5" />
              System Performance
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-300">Disk Usage</span>
                  <Badge variant="neon" className="text-xs">
                    {metrics.diskUsage.toFixed(1)}%
                  </Badge>
                </div>
                <Progress 
                  value={metrics.diskUsage} 
                  variant="neon" 
                  glowColor="blue" 
                  className="h-2" 
                />
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-300">Model Accuracy</span>
                  <Badge variant="rainbow" className="text-xs">
                    {metrics.modelAccuracy.toFixed(1)}%
                  </Badge>
                </div>
                <Progress 
                  value={metrics.modelAccuracy} 
                  variant="rainbow" 
                  className="h-2" 
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Threats Summary */}
      <motion.div
        whileHover={{ scale: 1.05 }}
        transition={{ type: "spring", stiffness: 300 }}
        className="md:col-span-2"
      >
        <Card variant="glass" className="neon-glow-red">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Activity className="h-5 w-5 text-red-400" />
              Security Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-red-400">{metrics.threatsBlocked}</div>
                <p className="text-sm text-gray-300">Threats Blocked Today</p>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-green-400">{metrics.activeConnections}</div>
                <p className="text-sm text-gray-300">Active Connections</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
};