// API client for real network monitoring data
const API_BASE_URL = 'http://localhost:3001/api';

export interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  networkLoad: number;
  diskUsage: number;
  gpuUsage: number;
  threatsBlocked: number;
  activeConnections: number;
  modelAccuracy: number;
}

export interface NetworkStatus {
  isMonitoring: boolean;
  activeConnections: number;
  packetsPerSecond: number;
  bytesPerSecond: number;
  networkInterfaces: string[];
  selectedInterface?: string;
}

export interface ThreatEvent {
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

export class NetworkAPI {
  private static instance: NetworkAPI;
  private socket: any = null;
  private listeners: Map<string, Function[]> = new Map();

  private constructor() {
    this.initializeSocket();
  }

  static getInstance(): NetworkAPI {
    if (!NetworkAPI.instance) {
      NetworkAPI.instance = new NetworkAPI();
    }
    return NetworkAPI.instance;
  }

  private async initializeSocket() {
    try {
      // Dynamic import of socket.io-client
      const { io } = await import('socket.io-client');
      this.socket = io('http://localhost:3001', {
        autoConnect: true,
        transports: ['websocket', 'polling']
      });

      this.socket.on('connect', () => {
        console.log('Connected to NetProtect API Server');
      });

      this.socket.on('disconnect', () => {
        console.log('Disconnected from NetProtect API Server');
      });

      this.socket.on('networkData', (data: any) => {
        this.emit('networkData', data);
      });

      this.socket.on('threatDetection', (threat: ThreatEvent) => {
        this.emit('threatDetection', threat);
      });

      this.socket.on('systemMetrics', (metrics: SystemMetrics) => {
        this.emit('systemMetrics', metrics);
      });

    } catch (error) {
      console.error('Failed to initialize socket connection:', error);
    }
  }

  private emit(event: string, data: any) {
    const listeners = this.listeners.get(event) || [];
    listeners.forEach(listener => listener(data));
  }

  on(event: string, listener: Function) {
    const listeners = this.listeners.get(event) || [];
    listeners.push(listener);
    this.listeners.set(event, listeners);
  }

  off(event: string, listener: Function) {
    const listeners = this.listeners.get(event) || [];
    const index = listeners.indexOf(listener);
    if (index > -1) {
      listeners.splice(index, 1);
      this.listeners.set(event, listeners);
    }
  }

  async getNetworkStatus(): Promise<NetworkStatus> {
    try {
      const response = await fetch(`${API_BASE_URL}/network/status`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Failed to fetch network status:', error);
      // Return fallback data if API is unavailable
      return {
        isMonitoring: false,
        activeConnections: 0,
        packetsPerSecond: 0,
        bytesPerSecond: 0,
        networkInterfaces: ['Ethernet', 'Wi-Fi'],
        selectedInterface: 'Ethernet'
      };
    }
  }

  async getCurrentNetworkData(): Promise<SystemMetrics> {
    try {
      const response = await fetch(`${API_BASE_URL}/network/current`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Failed to fetch current network data:', error);
      // Return fallback data if API is unavailable
      return {
        cpuUsage: 0,
        memoryUsage: 0,
        networkLoad: 0,
        diskUsage: 0,
        gpuUsage: 0,
        threatsBlocked: 0,
        activeConnections: 0,
        modelAccuracy: 0
      };
    }
  }

  async getMLPerformanceMetrics(): Promise<{
    accuracy: number;
    latency: number;
    throughput: number;
    confidence: number;
    predictionsPerSecond: number;
    processingTime: number;
    totalPredictions: number;
    anomaliesDetected: number;
  }> {
    try {
      const response = await fetch(`${API_BASE_URL}/ml/performance`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      return data.performance;
    } catch (error) {
      console.error('Failed to fetch ML performance metrics:', error);
      // Return fallback data if API is unavailable
      return {
        accuracy: 0.0,
        latency: 0,
        throughput: 0,
        confidence: 0.0,
        predictionsPerSecond: 0,
        processingTime: 0,
        totalPredictions: 0,
        anomaliesDetected: 0
      };
    }
  }

  async getTrafficData(): Promise<{
    totalBytes: number;
    incomingBytes: number;
    outgoingBytes: number;
    packetsPerSecond: number;
    connectionsActive: number;
    bandwidthUtilization: number;
  }> {
    try {
      const response = await fetch(`${API_BASE_URL}/network/traffic`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      return data.traffic;
    } catch (error) {
      console.error('Failed to fetch traffic data:', error);
      // Return fallback data if API is unavailable
      return {
        totalBytes: 0,
        incomingBytes: 0,
        outgoingBytes: 0,
        packetsPerSecond: 0,
        connectionsActive: 0,
        bandwidthUtilization: 0
      };
    }
  }

  async getNetworkTopology(): Promise<{
    nodes: Array<{
      id: string;
      ip: string;
      type: 'router' | 'server' | 'workstation' | 'unknown';
      status: 'normal' | 'suspicious' | 'threat';
      connections: number;
      lastSeen: number;
      riskScore: number;
    }>;
    totalNodes: number;
    activeConnections: number;
  }> {
    try {
      const response = await fetch(`${API_BASE_URL}/network/topology`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      return {
        nodes: data.nodes || [],
        totalNodes: data.totalNodes || 0,
        activeConnections: data.activeConnections || 0
      };
    } catch (error) {
      console.error('Failed to fetch network topology:', error);
      // Return fallback data if API is unavailable
      return {
        nodes: [],
        totalNodes: 0,
        activeConnections: 0
      };
    }
  }

  async getCurrentThreats(): Promise<{
    threats: ThreatEvent[];
    totalThreats: number;
    activeMonitoring: boolean;
  }> {
    try {
      const response = await fetch(`${API_BASE_URL}/threats/current`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      return {
        threats: data.threats || [],
        totalThreats: data.totalThreats || 0,
        activeMonitoring: data.activeMonitoring || false
      };
    } catch (error) {
      console.error('Failed to fetch current threats:', error);
      // Return fallback data if API is unavailable
      return {
        threats: [],
        totalThreats: 0,
        activeMonitoring: false
      };
    }
  }

  async startNetworkMonitoring(networkInterface?: string): Promise<void> {
    try {
      const response = await fetch(`${API_BASE_URL}/network/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ interface: networkInterface }),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Failed to start network monitoring:', error);
      throw error;
    }
  }

  async stopNetworkMonitoring(): Promise<void> {
    try {
      const response = await fetch(`${API_BASE_URL}/network/stop`, {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Failed to stop network monitoring:', error);
      throw error;
    }
  }

  async getAvailableNetworkInterfaces(): Promise<string[]> {
    try {
      // Use netsh command to get network interfaces like Wireshark
      const response = await fetch(`${API_BASE_URL}/network/interfaces`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      return data.interfaces || ['Ethernet', 'Wi-Fi'];
    } catch (error) {
      console.error('Failed to fetch network interfaces:', error);
      return ['Ethernet', 'Wi-Fi']; // Fallback interfaces
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
    }
  }
}

export const networkAPI = NetworkAPI.getInstance();