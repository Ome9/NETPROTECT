export interface RealNetworkConnection {
    protocol: string;
    localAddress: string;
    localPort: string;
    remoteAddress: string;
    remotePort: string;
    state: string;
    processName?: string;
    pid?: number;
}
export interface RealSystemMetrics {
    timestamp: string;
    cpuUsage: number;
    memoryUsage: number;
    diskUsage: number;
    networkInterfaces: {
        name: string;
        bytesReceived: number;
        bytesSent: number;
        packetsReceived: number;
        packetsSent: number;
    }[];
    activeConnections: RealNetworkConnection[];
    networkStats: {
        totalConnections: number;
        tcpConnections: number;
        udpConnections: number;
        listeningPorts: number;
    };
}
export declare class RealNetworkMonitor {
    private isMonitoring;
    private monitoringInterval;
    private previousNetworkStats;
    constructor();
    startRealTimeMonitoring(intervalMs?: number): Promise<void>;
    stopMonitoring(): void;
    getRealSystemMetrics(): Promise<RealSystemMetrics>;
    private getRealCpuUsage;
    private getRealMemoryUsage;
    private getRealDiskUsage;
    private getRealNetworkInterfaceStats;
    private getRealActiveConnections;
    private getRealNetworkStats;
    private collectInitialNetworkStats;
    private onMetricsCollected;
    private getProcessNameByPid;
    getNetworkTrafficByProcess(): Promise<any[]>;
}
//# sourceMappingURL=RealNetworkMonitor.d.ts.map