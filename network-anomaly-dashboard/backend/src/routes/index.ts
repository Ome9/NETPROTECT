import { Router, Request, Response } from 'express';
import { NetworkMonitor } from '../services/NetworkMonitor';
import { AnomalyDetectionService } from '../services/AnomalyDetectionService';

const router = Router();
const networkMonitor = new NetworkMonitor();
const anomalyService = new AnomalyDetectionService(process.env.ML_MODEL_URL || 'http://localhost:5000');

// Network status endpoint
router.get('/network/status', async (req: Request, res: Response) => {
  try {
    const systemMetrics = await networkMonitor.getSystemMetrics();
    const isMonitoring = networkMonitor.isCurrentlyMonitoring();
    
    res.json({
      isMonitoring,
      systemMetrics,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error getting network status:', error);
    res.status(500).json({ error: 'Failed to get network status' });
  }
});

// Start network monitoring
router.post('/network/start', async (req: Request, res: Response) => {
  try {
    const { logFilePath } = req.body;
    networkMonitor.startMonitoring(logFilePath);
    
    res.json({ 
      success: true, 
      message: 'Network monitoring started',
      logFilePath: logFilePath || null
    });
  } catch (error) {
    console.error('Error starting network monitoring:', error);
    res.status(500).json({ error: 'Failed to start network monitoring' });
  }
});

// Stop network monitoring
router.post('/network/stop', async (req: Request, res: Response) => {
  try {
    networkMonitor.stopMonitoring();
    
    res.json({ 
      success: true, 
      message: 'Network monitoring stopped'
    });
  } catch (error) {
    console.error('Error stopping network monitoring:', error);
    res.status(500).json({ error: 'Failed to stop network monitoring' });
  }
});

// Get current network data
router.get('/network/data', async (req: Request, res: Response) => {
  try {
    const networkData = await networkMonitor.getCurrentNetworkData();
    
    res.json({
      data: networkData,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error getting network data:', error);
    res.status(500).json({ error: 'Failed to get network data' });
  }
});

// Get historical network data
router.get('/network/history', async (req: Request, res: Response) => {
  try {
    const minutes = parseInt(req.query.minutes as string) || 60;
    const historicalData = networkMonitor.getHistoricalData(minutes);
    
    res.json({
      data: historicalData,
      minutes,
      count: historicalData.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error getting historical network data:', error);
    res.status(500).json({ error: 'Failed to get historical data' });
  }
});

// Get network adapters
router.get('/network/adapters', async (req: Request, res: Response) => {
  try {
    const adapters = await networkMonitor.getNetworkAdapters();
    
    res.json({
      adapters,
      count: adapters.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error getting network adapters:', error);
    res.status(500).json({ error: 'Failed to get network adapters' });
  }
});

// Anomaly detection endpoint
router.post('/anomaly/detect', async (req: Request, res: Response) => {
  try {
    const { networkData } = req.body;
    
    if (!networkData) {
      return res.status(400).json({ error: 'Network data is required' });
    }
    
    const results = await anomalyService.detectAnomalies(networkData);
    
    res.json({
      results,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error detecting anomalies:', error);
    res.status(500).json({ error: 'Failed to detect anomalies' });
  }
});

// ML model status
router.get('/anomaly/model-status', async (req: Request, res: Response) => {
  try {
    const status = await anomalyService.getModelStatus();
    
    res.json({
      modelStatus: status,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error getting model status:', error);
    res.status(500).json({ error: 'Failed to get model status' });
  }
});

export default router;