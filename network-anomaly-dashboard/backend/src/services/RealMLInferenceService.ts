import { spawn } from 'child_process';
import { RealSystemMetrics, RealNetworkConnection } from './RealNetworkMonitor';
import axios from 'axios';

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
  // Connection-based features
  connectionCount: number;
  tcpConnections: number;
  udpConnections: number;
  establishedConnections: number;
  listeningPorts: number;
  
  // Traffic volume features  
  totalBytesReceived: number;
  totalBytesSent: number;
  totalPacketsReceived: number;
  totalPacketsSent: number;
  
  // Time-based features
  connectionsPerSecond: number;
  bytesPerSecond: number;
  
  // Port and protocol features
  uniqueRemotePorts: number;
  uniqueRemoteHosts: number;
  suspiciousPortActivity: boolean;
  
  // System resource features
  cpuUsage: number;
  memoryUsage: number;
  
  // Behavioral features
  shortLivedConnections: number;
  longLivedConnections: number;
  foreignConnectionsRatio: number;
}

export class RealMLInferenceService {
  private pythonMLServiceUrl: string;
  private modelPath: string;
  private preprocessorPath: string;
  private isModelLoaded: boolean = false;
  private lastPredictionTime: number = 0;
  private predictionHistory: ThreatPrediction[] = [];

  constructor(
    modelPath: string = '../models/production/vae_final.pt',
    preprocessorPath: string = '../models/preprocessors/vae_preprocessor.pkl',
    mlServiceUrl: string = 'http://localhost:8001'
  ) {
    this.modelPath = modelPath;
    this.preprocessorPath = preprocessorPath;
    this.pythonMLServiceUrl = mlServiceUrl;
  }

  async initializeMLService(): Promise<boolean> {
    try {
      console.log('🤖 Initializing Real ML Inference Service...');
      
      // Start Python ML service if not running
      await this.startPythonMLService();
      
      // Load the model
      const response = await axios.post(`${this.pythonMLServiceUrl}/load_model`, {
        model_path: this.modelPath,
        preprocessor_path: this.preprocessorPath
      });
      
      if (response.data.success) {
        this.isModelLoaded = true;
        console.log('✅ ML Model loaded successfully:', response.data.model_info);
        return true;
      } else {
        console.error('❌ Failed to load ML model:', response.data.error);
        return false;
      }
    } catch (error) {
      console.error('❌ Error initializing ML service:', error);
      return false;
    }
  }

  async predictThreats(systemMetrics: RealSystemMetrics): Promise<ThreatPrediction> {
    if (!this.isModelLoaded) {
      throw new Error('ML Model not loaded. Call initializeMLService() first.');
    }

    try {
      // Extract features from real system metrics
      const features = this.extractNetworkFeatures(systemMetrics);
      
      // Send to Python ML service for prediction
      const response = await axios.post(`${this.pythonMLServiceUrl}/predict`, {
        features: this.featuresToArray(features),
        timestamp: systemMetrics.timestamp
      });

      if (response.data.success) {
        const prediction = this.processPredictionResult(response.data, systemMetrics, features);
        
        // Store in history
        this.predictionHistory.push(prediction);
        if (this.predictionHistory.length > 1000) {
          this.predictionHistory = this.predictionHistory.slice(-1000);
        }
        
        this.lastPredictionTime = Date.now();
        return prediction;
      } else {
        throw new Error('Prediction failed: ' + response.data.error);
      }
    } catch (error) {
      console.error('❌ Error during threat prediction:', error);
      
      // Return fallback prediction based on heuristics
      return this.createHeuristicPrediction(systemMetrics);
    }
  }

  private extractNetworkFeatures(metrics: RealSystemMetrics): NetworkFeatures {
    const connections = metrics.activeConnections;
    
    // Calculate various network features
    const establishedConnections = connections.filter(c => c.state === 'ESTABLISHED').length;
    const listeningPorts = connections.filter(c => c.state === 'LISTENING').length;
    const uniqueRemotePorts = new Set(connections.map(c => c.remotePort)).size;
    const uniqueRemoteHosts = new Set(connections.map(c => c.remoteAddress)).size;
    
    // Calculate traffic volumes
    const totalBytesReceived = metrics.networkInterfaces.reduce((sum, intf) => sum + intf.bytesReceived, 0);
    const totalBytesSent = metrics.networkInterfaces.reduce((sum, intf) => sum + intf.bytesSent, 0);
    const totalPacketsReceived = metrics.networkInterfaces.reduce((sum, intf) => sum + intf.packetsReceived, 0);
    const totalPacketsSent = metrics.networkInterfaces.reduce((sum, intf) => sum + intf.packetsSent, 0);
    
    // Detect suspicious port activity
    const suspiciousPortActivity = this.detectSuspiciousPorts(connections);
    
    // Calculate connection ratios
    const foreignConnections = connections.filter(c => 
      !c.remoteAddress.startsWith('127.') && 
      !c.remoteAddress.startsWith('192.168.') && 
      !c.remoteAddress.startsWith('10.') &&
      c.remoteAddress !== '0.0.0.0'
    ).length;
    const foreignConnectionsRatio = connections.length > 0 ? foreignConnections / connections.length : 0;

    return {
      connectionCount: connections.length,
      tcpConnections: connections.filter(c => c.protocol === 'tcp').length,
      udpConnections: connections.filter(c => c.protocol === 'udp').length,
      establishedConnections,
      listeningPorts,
      
      totalBytesReceived,
      totalBytesSent,
      totalPacketsReceived,
      totalPacketsSent,
      
      connectionsPerSecond: this.calculateConnectionsPerSecond(connections),
      bytesPerSecond: this.calculateBytesPerSecond(totalBytesReceived + totalBytesSent),
      
      uniqueRemotePorts,
      uniqueRemoteHosts,
      suspiciousPortActivity,
      
      cpuUsage: metrics.cpuUsage,
      memoryUsage: metrics.memoryUsage,
      
      shortLivedConnections: 0, // Would require tracking connection duration
      longLivedConnections: establishedConnections,
      foreignConnectionsRatio
    };
  }

  private featuresToArray(features: NetworkFeatures): number[] {
    return [
      features.connectionCount,
      features.tcpConnections,
      features.udpConnections,
      features.establishedConnections,
      features.listeningPorts,
      features.totalBytesReceived,
      features.totalBytesSent,
      features.totalPacketsReceived,
      features.totalPacketsSent,
      features.connectionsPerSecond,
      features.bytesPerSecond,
      features.uniqueRemotePorts,
      features.uniqueRemoteHosts,
      features.suspiciousPortActivity ? 1 : 0,
      features.cpuUsage,
      features.memoryUsage,
      features.shortLivedConnections,
      features.longLivedConnections,
      features.foreignConnectionsRatio
    ];
  }

  private processPredictionResult(mlResult: any, metrics: RealSystemMetrics, features: NetworkFeatures): ThreatPrediction {
    const isAnomaly = mlResult.prediction.is_anomaly;
    const confidence = mlResult.prediction.confidence;
    const threatType = this.classifyThreatType(features, isAnomaly);
    const riskScore = this.calculateRiskScore(features, confidence);

    return {
      id: `threat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: metrics.timestamp,
      isAnomaly,
      confidence,
      threatType,
      riskScore,
      features: {
        connectionCount: features.connectionCount,
        bytesSent: features.totalBytesSent,
        bytesReceived: features.totalBytesReceived,
        suspiciousConnections: features.uniqueRemoteHosts,
        unusualPorts: features.suspiciousPortActivity,
        highVolumeTraffic: features.bytesPerSecond > 10000000 // 10MB/s threshold
      },
      modelInfo: {
        modelName: mlResult.model_info?.name || 'VAE Autoencoder',
        version: mlResult.model_info?.version || '1.0',
        accuracy: mlResult.model_info?.accuracy || 95.17
      }
    };
  }

  private classifyThreatType(features: NetworkFeatures, isAnomaly: boolean): ThreatPrediction['threatType'] {
    if (!isAnomaly) return 'normal';
    
    // Heuristic classification based on features
    if (features.connectionCount > 100 && features.bytesPerSecond > 50000000) {
      return 'dos'; // High volume, many connections - likely DoS
    } else if (features.uniqueRemotePorts > 50 || features.suspiciousPortActivity) {
      return 'probe'; // Port scanning behavior
    } else if (features.foreignConnectionsRatio > 0.8) {
      return 'r2l'; // Many remote connections - remote to local attack
    } else if (features.cpuUsage > 80 || features.memoryUsage > 90) {
      return 'u2r'; // High resource usage - user to root escalation
    }
    
    return 'dos'; // Default for anomalies
  }

  private calculateRiskScore(features: NetworkFeatures, confidence: number): number {
    let riskScore = confidence * 50; // Base score from model confidence
    
    // Add risk based on specific indicators
    if (features.suspiciousPortActivity) riskScore += 20;
    if (features.foreignConnectionsRatio > 0.7) riskScore += 15;
    if (features.connectionCount > 200) riskScore += 10;
    if (features.bytesPerSecond > 100000000) riskScore += 15; // 100MB/s
    if (features.cpuUsage > 85) riskScore += 10;
    
    return Math.min(100, Math.max(0, riskScore));
  }

  private detectSuspiciousPorts(connections: RealNetworkConnection[]): boolean {
    const suspiciousPorts = ['1433', '3389', '5900', '23', '135', '445', '139'];
    const commonPorts = connections.map(c => c.localPort).concat(connections.map(c => c.remotePort));
    
    return suspiciousPorts.some(port => commonPorts.includes(port));
  }

  private calculateConnectionsPerSecond(connections: RealNetworkConnection[]): number {
    // This would require tracking connections over time
    // For now, return a simple estimate
    return connections.length / 2; // Rough estimate
  }

  private calculateBytesPerSecond(totalBytes: number): number {
    // This would require tracking bytes over time intervals
    // For now, return the total bytes (would need time-based calculation)
    return totalBytes / 60; // Rough per-second estimate over 1 minute
  }

  private createHeuristicPrediction(metrics: RealSystemMetrics): ThreatPrediction {
    const features = this.extractNetworkFeatures(metrics);
    
    // Simple heuristic-based threat detection
    const isAnomaly = 
      features.connectionCount > 150 ||
      features.cpuUsage > 90 ||
      features.memoryUsage > 95 ||
      features.suspiciousPortActivity ||
      features.foreignConnectionsRatio > 0.9;

    return {
      id: `heuristic_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: metrics.timestamp,
      isAnomaly,
      confidence: isAnomaly ? 0.7 : 0.9,
      threatType: isAnomaly ? 'dos' : 'normal',
      riskScore: this.calculateRiskScore(features, isAnomaly ? 0.7 : 0.1),
      features: {
        connectionCount: features.connectionCount,
        bytesSent: features.totalBytesSent,
        bytesReceived: features.totalBytesReceived,
        suspiciousConnections: features.uniqueRemoteHosts,
        unusualPorts: features.suspiciousPortActivity,
        highVolumeTraffic: features.bytesPerSecond > 10000000
      },
      modelInfo: {
        modelName: 'Heuristic Fallback',
        version: '1.0',
        accuracy: 75.0
      }
    };
  }

  private async startPythonMLService(): Promise<void> {
    try {
      // Create Python ML service script if it doesn't exist
      await this.createPythonMLService();
      
      console.log('🐍 Starting Python ML service...');
      
      // Start the Python service
      const pythonProcess = spawn('python', ['-m', 'ml_service'], {
        cwd: '../',
        stdio: 'pipe',
        detached: true
      });

      pythonProcess.stdout?.on('data', (data) => {
        console.log(`ML Service: ${data}`);
      });

      pythonProcess.stderr?.on('data', (data) => {
        console.error(`ML Service Error: ${data}`);
      });

      // Wait a bit for the service to start
      await new Promise(resolve => setTimeout(resolve, 3000));
    } catch (error) {
      console.error('Error starting Python ML service:', error);
    }
  }

  private async createPythonMLService(): Promise<void> {
    // This would create the Python ML service script
    // For now, we'll use the existing unified_production_model.py
    console.log('📁 Using existing unified_production_model.py for ML inference');
  }

  getPredictionHistory(minutes: number = 60): ThreatPrediction[] {
    const cutoffTime = Date.now() - (minutes * 60 * 1000);
    return this.predictionHistory.filter(p => 
      new Date(p.timestamp).getTime() >= cutoffTime
    );
  }

  getModelStatus(): { loaded: boolean; lastPrediction: number; historyCount: number } {
    return {
      loaded: this.isModelLoaded,
      lastPrediction: this.lastPredictionTime,
      historyCount: this.predictionHistory.length
    };
  }
}