import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

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

interface RealTimeChartProps {
  data: NetworkData[];
}

export const RealTimeChart: React.FC<RealTimeChartProps> = ({ data }) => {
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    });
  };

  // const formatBytes = (bytes: number) => {
  //   if (bytes === 0) return '0';
  //   const k = 1024;
  //   const sizes = ['B', 'KB', 'MB', 'GB'];
  //   const i = Math.floor(Math.log(bytes) / Math.log(k));
  //   return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + sizes[i];
  // };

  // Transform data for the chart
  const chartData = data.slice(-20).map(item => ({
    time: formatTime(item.timestamp),
    connections: item.connections,
    bytesIn: item.bytesIn / (1024 * 1024), // Convert to MB
    bytesOut: item.bytesOut / (1024 * 1024), // Convert to MB
    packetsIn: item.packetsIn,
    packetsOut: item.packetsOut,
    totalTraffic: (item.bytesIn + item.bytesOut) / (1024 * 1024)
  }));

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        <div className="text-center">
          <div className="text-4xl mb-2">📊</div>
          <p>Waiting for network data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Network Traffic Chart */}
      <div className="h-64">
        <h4 className="text-sm font-semibold text-gray-300 mb-3">Network Traffic (MB/s)</h4>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="time" 
              stroke="#9CA3AF" 
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis 
              stroke="#9CA3AF" 
              fontSize={12}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${value.toFixed(1)}`}
            />
            <RechartsTooltip 
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#F3F4F6'
              }}
              formatter={(value: number, name) => [
                `${typeof value === 'number' ? value.toFixed(2) : value} MB`,
                name === 'bytesIn' ? 'Inbound' : name === 'bytesOut' ? 'Outbound' : 'Total'
              ]}
            />
            <Area
              type="monotone"
              dataKey="bytesIn"
              stackId="1"
              stroke="#10B981"
              fill="#10B981"
              fillOpacity={0.6}
            />
            <Area
              type="monotone"
              dataKey="bytesOut"
              stackId="1"
              stroke="#F59E0B"
              fill="#F59E0B"
              fillOpacity={0.6}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Connections Chart */}
      <div className="h-48">
        <h4 className="text-sm font-semibold text-gray-300 mb-3">Active Connections</h4>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="time" 
              stroke="#9CA3AF" 
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis 
              stroke="#9CA3AF" 
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <RechartsTooltip 
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#F3F4F6'
              }}
              formatter={(value: number) => [`${value}`, 'Connections']}
            />
            <Line
              type="monotone"
              dataKey="connections"
              stroke="#3B82F6"
              strokeWidth={2}
              dot={{ fill: '#3B82F6', strokeWidth: 2, r: 3 }}
              activeDot={{ r: 5, stroke: '#3B82F6', strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};