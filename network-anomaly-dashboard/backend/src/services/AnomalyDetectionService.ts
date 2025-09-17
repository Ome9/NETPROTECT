import axios from 'axios';

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

export class AnomalyDetectionService {
  private modelUrl: string;

  constructor(modelUrl: string) {
    this.modelUrl = modelUrl;
  }

  async detectAnomalies(networkData: NetworkData | NetworkData[]): Promise<AnomalyResult[]> {
    try {
      const response = await axios.post(`${this.modelUrl}/predict`, {
        data: Array.isArray(networkData) ? networkData : [networkData]
      }, {
        timeout: 5000,
        headers: {
          'Content-Type': 'application/json'
        }
      });

      return response.data.results || [];
    } catch (error) {
      console.error('Error calling ML model:', error);
      // Return mock data for development
      return this.generateMockResults(networkData);
    }
  }

  private generateMockResults(networkData: NetworkData | NetworkData[]): AnomalyResult[] {
    const dataArray = Array.isArray(networkData) ? networkData : [networkData];
    
    return dataArray.map(() => ({
      isAnomaly: Math.random() > 0.8, // 20% chance of anomaly
      confidence: Math.random(),
      score: Math.random() * 2,
      datasetType: 'nsl-kdd',
      features: 41
    }));
  }

  async getModelStatus(): Promise<{ available: boolean; models: string[] }> {
    try {
      const response = await axios.get(`${this.modelUrl}/status`);
      return response.data;
    } catch (error) {
      console.error('Error checking model status:', error);
      return { available: false, models: [] };
    }
  }
}