'use client';

import React from 'react';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';

interface NetworkData {
  timestamp: string;
  connections: number;
  bytesIn: number;
  bytesOut: number;
  packetsIn: number;
  packetsOut: number;
  protocols: Record<string, number>;
  ports: Record<string, number>;
}

interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  networkInterfaces: Array<{
    name: string;
    type: string;
    status: string;
    address: string;
  }>;
}

interface NetworkStatsProps {
  networkData: NetworkData | null;
  systemMetrics: SystemMetrics | null;
}

export const NetworkStats: React.FC<NetworkStatsProps> = ({ 
  networkData = null, 
  systemMetrics = null 
}) => {
  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatNumber = (num: number): string => {
    return num.toLocaleString();
  };

  return (
    <div className="space-y-6">
      {/* System Metrics */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-gray-200">System Performance</h3>
        
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">CPU Usage</span>
            <span className="text-gray-200">{systemMetrics?.cpuUsage?.toFixed(1) ?? '0'}%</span>
          </div>
          <Progress 
            value={systemMetrics?.cpuUsage ?? 0} 
            className="h-2"
          />
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Memory Usage</span>
            <span className="text-gray-200">{systemMetrics?.memoryUsage?.toFixed(1) ?? '0'}%</span>
          </div>
          <Progress 
            value={systemMetrics?.memoryUsage ?? 0} 
            className="h-2"
          />
        </div>
      </div>

      <Separator className="bg-gray-700" />

      {/* Network Data */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-gray-200">Network Traffic</h3>
        
        {networkData ? (
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Connections</span>
              <div className="text-lg font-semibold text-blue-400">
                {formatNumber(networkData.connections)}
              </div>
            </div>
            
            <div>
              <span className="text-gray-400">Bytes In</span>
              <div className="text-lg font-semibold text-green-400">
                {formatBytes(networkData.bytesIn)}
              </div>
            </div>
            
            <div>
              <span className="text-gray-400">Bytes Out</span>
              <div className="text-lg font-semibold text-yellow-400">
                {formatBytes(networkData.bytesOut)}
              </div>
            </div>
            
            <div>
              <span className="text-gray-400">Packets</span>
              <div className="text-lg font-semibold text-purple-400">
                {formatNumber(networkData.packetsIn + networkData.packetsOut)}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-gray-400 text-sm">No network data available</div>
        )}
      </div>

      <Separator className="bg-gray-700" />

      {/* Protocol Distribution */}
      {networkData?.protocols && Object.keys(networkData.protocols).length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-gray-200">Protocol Distribution</h3>
          <div className="space-y-2">
            {Object.entries(networkData.protocols).map(([protocol, count]) => (
              <div key={protocol} className="flex justify-between items-center">
                <Badge variant="outline" className="text-xs">
                  {protocol.toUpperCase()}
                </Badge>
                <span className="text-sm text-gray-300">{count}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <Separator className="bg-gray-700" />

      {/* Network Interfaces */}
      {systemMetrics?.networkInterfaces && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-gray-200">Network Interfaces</h3>
          <div className="space-y-2">
            {systemMetrics.networkInterfaces.slice(0, 3).map((iface, index) => (
              <div key={index} className="space-y-1">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-300 truncate">{iface.name}</span>
                  <Badge 
                    variant={iface.status === 'Connected' ? 'default' : 'secondary'}
                    className="text-xs"
                  >
                    {iface.status}
                  </Badge>
                </div>
                <div className="text-xs text-gray-400 truncate">{iface.address}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Active Ports */}
      {networkData?.ports && Object.keys(networkData.ports).length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-gray-200">Active Ports</h3>
          <div className="space-y-1">
            {Object.entries(networkData.ports)
              .sort(([,a], [,b]) => b - a)
              .slice(0, 5)
              .map(([port, count]) => (
              <div key={port} className="flex justify-between items-center text-sm">
                <span className="text-gray-400">Port {port}</span>
                <span className="text-gray-200">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};