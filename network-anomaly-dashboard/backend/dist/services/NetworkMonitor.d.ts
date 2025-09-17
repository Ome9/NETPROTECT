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
    networkLoad: number;
    diskUsage: number;
    gpuUsage: number;
    networkInterfaces: NetworkAdapterInfo[];
}
export declare class NetworkMonitor {
    private isMonitoring;
    private networkData;
    private logFilePath;
    constructor();
    getNetworkInterfaces(): Promise<string[]>;
    getNetworkAdapters(): Promise<NetworkAdapterInfo[]>;
    getSystemMetrics(): Promise<SystemMetrics>;
    private getCpuUsage;
    private getFallbackCpuUsage;
    private getMemoryUsage;
    private getDiskUsage;
    private getGpuUsage;
    getSystemMetricsForAPI(): Promise<SystemMetrics & {
        threatsBlocked: number;
        activeConnections: number;
        modelAccuracy: number;
    }>;
    getCurrentNetworkData(): Promise<NetworkData>;
    private getRealNetworkStats;
    private estimateNetworkActivity;
    getHistoricalData(minutes?: number): NetworkData[];
    startMonitoring(logFilePath?: string): void;
    stopMonitoring(): void;
    private startLogFileMonitoring;
    private processLogLine;
    getTrafficData(): Promise<{
        totalBytes: number;
        incomingBytes: number;
        outgoingBytes: number;
        packetsPerSecond: number;
        connectionsActive: number;
        bandwidthUtilization: number;
    }>;
    isCurrentlyMonitoring(): boolean;
}
//# sourceMappingURL=NetworkMonitor.d.ts.map