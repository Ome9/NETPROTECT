import { RealSystemMetrics } from './RealNetworkMonitor';
export interface ThreatPrediction {
    id: string;
    timestamp: string;
    isAnomaly: boolean;
    confidence: number;
    threatType: 'normal' | 'dos' | 'probe' | 'r2l' | 'u2r';
    riskScore: number;
    features: {
        connectionCount: number;
        bytesSent: number;
        bytesReceived: number;
        suspiciousConnections: number;
        unusualPorts: boolean;
        highVolumeTraffic: boolean;
    };
    modelInfo: {
        modelName: string;
        version: string;
        accuracy: number;
    };
}
export interface NetworkFeatures {
    connectionCount: number;
    tcpConnections: number;
    udpConnections: number;
    establishedConnections: number;
    listeningPorts: number;
    totalBytesReceived: number;
    totalBytesSent: number;
    totalPacketsReceived: number;
    totalPacketsSent: number;
    connectionsPerSecond: number;
    bytesPerSecond: number;
    uniqueRemotePorts: number;
    uniqueRemoteHosts: number;
    suspiciousPortActivity: boolean;
    cpuUsage: number;
    memoryUsage: number;
    shortLivedConnections: number;
    longLivedConnections: number;
    foreignConnectionsRatio: number;
}
export declare class RealMLInferenceService {
    private pythonMLServiceUrl;
    private modelPath;
    private preprocessorPath;
    private isModelLoaded;
    private lastPredictionTime;
    private predictionHistory;
    constructor(modelPath?: string, preprocessorPath?: string, mlServiceUrl?: string);
    initializeMLService(): Promise<boolean>;
    predictThreats(systemMetrics: RealSystemMetrics): Promise<ThreatPrediction>;
    private extractNetworkFeatures;
    private featuresToArray;
    private processPredictionResult;
    private classifyThreatType;
    private calculateRiskScore;
    private detectSuspiciousPorts;
    private calculateConnectionsPerSecond;
    private calculateBytesPerSecond;
    private createHeuristicPrediction;
    private startPythonMLService;
    private createPythonMLService;
    getPredictionHistory(minutes?: number): ThreatPrediction[];
    getModelStatus(): {
        loaded: boolean;
        lastPrediction: number;
        historyCount: number;
    };
}
//# sourceMappingURL=RealMLInferenceService.d.ts.map