import { spawn } from 'child_process';
import * as os from 'os';
import { NetworkData } from './AnomalyDetectionService';

export interface NetworkAdapterInfo {
  name: string;
  type: string;
  status: string;
  address: string;
}

export interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  networkInterfaces: NetworkAdapterInfo[];
}

export class NetworkMonitor {
  private isMonitoring: boolean = false;
  private networkData: NetworkData[] = [];
  private logFilePath: string | null = null;

  constructor() {
    this.networkData = [];
  }

  async getNetworkAdapters(): Promise<NetworkAdapterInfo[]> {
    const networkInterfaces = os.networkInterfaces();
    const adapters: NetworkAdapterInfo[] = [];

    Object.keys(networkInterfaces).forEach(interfaceName => {
      const interfaceInfo = networkInterfaces[interfaceName];
      if (interfaceInfo) {
        interfaceInfo.forEach(info => {
          if (info.family === 'IPv4' && !info.internal) {
            adapters.push({
              name: interfaceName,
              type: 'Ethernet', // Could be enhanced to detect actual type
              status: 'Connected',
              address: info.address
            });
          }
        });
      }
    });

    return adapters;
  }

  async getSystemMetrics(): Promise<SystemMetrics> {
    const networkInterfaces = await this.getNetworkAdapters();
    
    return {
      cpuUsage: await this.getCpuUsage(),
      memoryUsage: this.getMemoryUsage(),
      networkInterfaces
    };
  }

  private async getCpuUsage(): Promise<number> {
    return new Promise((resolve) => {
      const startTime = process.hrtime();
      const startUsage = process.cpuUsage();

      setTimeout(() => {
        const currentUsage = process.cpuUsage(startUsage);
        const elapsedTime = process.hrtime(startTime);
        
        const elapsedMS = elapsedTime[0] * 1000 + elapsedTime[1] / 1e6;
        const cpuPercent = ((currentUsage.user + currentUsage.system) / 1000) / elapsedMS * 100;
        
        resolve(Math.round(cpuPercent * 100) / 100);
      }, 100);
    });
  }

  private getMemoryUsage(): number {
    const totalMemory = os.totalmem();
    const freeMemory = os.freemem();
    const usedMemory = totalMemory - freeMemory;
    
    return Math.round((usedMemory / totalMemory) * 100 * 100) / 100;
  }

  async getCurrentNetworkData(): Promise<NetworkData> {
    // Generate mock network data for now
    // In production, this would collect real network statistics
    const timestamp = new Date().toISOString();
    
    const networkData: NetworkData = {
      timestamp,
      connections: Math.floor(Math.random() * 100) + 10,
      bytesIn: Math.floor(Math.random() * 1000000) + 50000,
      bytesOut: Math.floor(Math.random() * 800000) + 30000,
      packetsIn: Math.floor(Math.random() * 5000) + 500,
      packetsOut: Math.floor(Math.random() * 4000) + 400,
      protocols: {
        TCP: Math.floor(Math.random() * 60) + 40,
        UDP: Math.floor(Math.random() * 30) + 10,
        ICMP: Math.floor(Math.random() * 10) + 5
      },
      ports: {
        '80': Math.floor(Math.random() * 30) + 10,
        '443': Math.floor(Math.random() * 40) + 20,
        '22': Math.floor(Math.random() * 10) + 2,
        '3389': Math.floor(Math.random() * 5) + 1
      }
    };

    // Store historical data
    this.networkData.push(networkData);
    
    // Keep only last 1000 entries
    if (this.networkData.length > 1000) {
      this.networkData = this.networkData.slice(-1000);
    }

    return networkData;
  }

  getHistoricalData(minutes: number = 60): NetworkData[] {
    const cutoffTime = new Date(Date.now() - minutes * 60 * 1000);
    
    return this.networkData.filter(data => 
      new Date(data.timestamp) >= cutoffTime
    );
  }

  startMonitoring(logFilePath?: string): void {
    this.isMonitoring = true;
    this.logFilePath = logFilePath || null;
    
    console.log('Network monitoring started');
    if (this.logFilePath) {
      console.log(`Monitoring log file: ${this.logFilePath}`);
      this.startLogFileMonitoring();
    }
  }

  stopMonitoring(): void {
    this.isMonitoring = false;
    console.log('Network monitoring stopped');
  }

  private startLogFileMonitoring(): void {
    if (!this.logFilePath) return;

    // On Windows, use PowerShell to tail log files
    const tailCommand = process.platform === 'win32' 
      ? 'powershell'
      : 'tail';
    
    const tailArgs = process.platform === 'win32'
      ? ['-Command', `Get-Content "${this.logFilePath}" -Wait -Tail 10`]
      : ['-f', this.logFilePath];

    try {
      const tailProcess = spawn(tailCommand, tailArgs);

      tailProcess.stdout.on('data', (data) => {
        if (this.isMonitoring) {
          const logLine = data.toString().trim();
          this.processLogLine(logLine);
        }
      });

      tailProcess.stderr.on('data', (data) => {
        console.error(`Log monitoring error: ${data}`);
      });

      tailProcess.on('close', (code) => {
        console.log(`Log monitoring process exited with code ${code}`);
      });
    } catch (error) {
      console.error('Failed to start log file monitoring:', error);
    }
  }

  private processLogLine(logLine: string): void {
    // Parse network log line and extract relevant information
    // This would be customized based on the actual log format
    console.log(`Processing log: ${logLine.substring(0, 100)}...`);
  }

  isCurrentlyMonitoring(): boolean {
    return this.isMonitoring;
  }
}