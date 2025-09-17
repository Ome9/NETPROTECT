// NetworkStats Component - Standalone TypeScript/HTML version
// This is a simplified version without React dependencies

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

export class NetworkStatsRenderer {
  private formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  private formatNumber(num: number): string {
    return num.toLocaleString();
  }

  public render(props: NetworkStatsProps): string {
    const { networkData, systemMetrics } = props;
    
    return `
      <div class="network-stats space-y-6">
        <!-- System Metrics -->
        <div class="space-y-3">
          <h3 class="text-sm font-semibold text-gray-200">System Performance</h3>
          
          <div class="space-y-2">
            <div class="flex justify-between text-sm">
              <span class="text-gray-400">CPU Usage</span>
              <span class="text-gray-200">${systemMetrics?.cpuUsage?.toFixed(1) ?? '0'}%</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-2">
              <div class="bg-blue-600 h-2 rounded-full transition-all" style="width: ${systemMetrics?.cpuUsage ?? 0}%"></div>
            </div>
          </div>

          <div class="space-y-2">
            <div class="flex justify-between text-sm">
              <span class="text-gray-400">Memory Usage</span>
              <span class="text-gray-200">${systemMetrics?.memoryUsage?.toFixed(1) ?? '0'}%</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-2">
              <div class="bg-blue-600 h-2 rounded-full transition-all" style="width: ${systemMetrics?.memoryUsage ?? 0}%"></div>
            </div>
          </div>
        </div>

        <div class="border-t border-gray-700"></div>

        <!-- Network Data -->
        <div class="space-y-3">
          <h3 class="text-sm font-semibold text-gray-200">Network Traffic</h3>
          
          ${networkData ? `
            <div class="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span class="text-gray-400">Connections</span>
                <div class="text-lg font-semibold text-blue-400">
                  ${this.formatNumber(networkData.connections)}
                </div>
              </div>
              
              <div>
                <span class="text-gray-400">Bytes In</span>
                <div class="text-lg font-semibold text-green-400">
                  ${this.formatBytes(networkData.bytesIn)}
                </div>
              </div>
              
              <div>
                <span class="text-gray-400">Bytes Out</span>
                <div class="text-lg font-semibold text-yellow-400">
                  ${this.formatBytes(networkData.bytesOut)}
                </div>
              </div>
              
              <div>
                <span class="text-gray-400">Packets</span>
                <div class="text-lg font-semibold text-purple-400">
                  ${this.formatNumber(networkData.packetsIn + networkData.packetsOut)}
                </div>
              </div>
            </div>
          ` : `
            <div class="text-gray-400 text-sm">No network data available</div>
          `}
        </div>

        ${networkData?.protocols && Object.keys(networkData.protocols).length > 0 ? `
          <div class="border-t border-gray-700"></div>
          <div class="space-y-3">
            <h3 class="text-sm font-semibold text-gray-200">Protocol Distribution</h3>
            <div class="space-y-2">
              ${Object.entries(networkData.protocols).map(([protocol, count]) => `
                <div class="flex justify-between items-center">
                  <span class="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium border border-gray-600 text-gray-300">
                    ${protocol.toUpperCase()}
                  </span>
                  <span class="text-sm text-gray-300">${count}%</span>
                </div>
              `).join('')}
            </div>
          </div>
        ` : ''}

        ${systemMetrics?.networkInterfaces ? `
          <div class="border-t border-gray-700"></div>
          <div class="space-y-3">
            <h3 class="text-sm font-semibold text-gray-200">Network Interfaces</h3>
            <div class="space-y-2">
              ${systemMetrics.networkInterfaces.slice(0, 3).map((iface, index) => `
                <div class="space-y-1">
                  <div class="flex justify-between items-center">
                    <span class="text-sm text-gray-300 truncate">${iface.name}</span>
                    <span class="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium ${
                      iface.status === 'Connected' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'
                    }">
                      ${iface.status}
                    </span>
                  </div>
                  <div class="text-xs text-gray-400 truncate">${iface.address}</div>
                </div>
              `).join('')}
            </div>
          </div>
        ` : ''}

        ${networkData?.ports && Object.keys(networkData.ports).length > 0 ? `
          <div class="border-t border-gray-700"></div>
          <div class="space-y-3">
            <h3 class="text-sm font-semibold text-gray-200">Active Ports</h3>
            <div class="space-y-1">
              ${Object.entries(networkData.ports)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 5)
                .map(([port, count]) => `
                  <div class="flex justify-between items-center text-sm">
                    <span class="text-gray-400">Port ${port}</span>
                    <span class="text-gray-200">${count}</span>
                  </div>
                `).join('')}
            </div>
          </div>
        ` : ''}
      </div>
    `;
  }
}

// Export types for use in other files
export type { NetworkData, SystemMetrics, NetworkStatsProps };