export interface NetworkData {
    timestamp: string;
    connections: number;
    bytesIn: number;
    bytesOut: number;
    packetsIn: number;
    packetsOut: number;
    protocols: Record<string, number>;
    ports: Record<string, number>;
    features?: number[];
}
export interface AnomalyResult {
    isAnomaly: boolean;
    confidence: number;
    score: number;
    datasetType: string;
    features: number;
}
export declare class AnomalyDetectionService {
    private modelUrl;
    constructor(modelUrl: string);
    detectAnomalies(networkData: NetworkData | NetworkData[]): Promise<AnomalyResult[]>;
    private generateMockResults;
    getModelStatus(): Promise<{
        available: boolean;
        models: string[];
    }>;
    getModelMetrics(): Promise<{
        accuracy: number;
        avgLatency: number;
        throughput: number;
        avgConfidence: number;
        predictionsPerSecond: number;
        avgProcessingTime: number;
        totalPredictions: number;
        anomaliesDetected: number;
    }>;
}
//# sourceMappingURL=AnomalyDetectionService.d.ts.map