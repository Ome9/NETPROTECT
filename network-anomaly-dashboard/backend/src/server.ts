import express from 'express';
import http from 'http';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import { Server } from 'socket.io';
import routes from './routes';
import { NetworkMonitor } from './services/NetworkMonitor';
import { AnomalyDetectionService } from './services/AnomalyDetectionService';

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: process.env.FRONTEND_URL || "http://localhost:3000",
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Initialize services
const networkMonitor = new NetworkMonitor();
const anomalyDetectionService = new AnomalyDetectionService(
  process.env.ML_MODEL_URL || 'http://localhost:8000'
);

// Initialize routes
app.use('/api', routes);

// Socket.IO for real-time communication
io.on('connection', (socket) => {
  console.log(`Client connected: ${socket.id}`);

  // Send real-time network data
  const networkDataInterval = setInterval(async () => {
    try {
      const networkData = await networkMonitor.getCurrentNetworkData();
      const anomalyResults = await anomalyDetectionService.detectAnomalies(networkData);
      
      socket.emit('network-data', {
        timestamp: new Date().toISOString(),
        data: networkData,
        anomalies: anomalyResults
      });
    } catch (error) {
      console.error('Error sending network data:', error);
    }
  }, 1000); // Send data every second

  // Handle configuration changes
  socket.on('configure-monitoring', (config) => {
    console.log('Monitoring configuration updated:', config);
    // Configuration would be handled here in a real implementation
  });

  // Handle manual anomaly detection requests
  socket.on('detect-anomalies', async (data) => {
    try {
      const results = await anomalyDetectionService.detectAnomalies(data.networkData);
      socket.emit('anomaly-results', {
        requestId: data.requestId,
        results
      });
    } catch (error) {
      socket.emit('anomaly-error', {
        requestId: data.requestId,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      });
    }
  });

  socket.on('disconnect', () => {
    console.log(`Client disconnected: ${socket.id}`);
    clearInterval(networkDataInterval);
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

const PORT = process.env.PORT || 5000;

server.listen(PORT, () => {
  console.log(`🚀 Network Anomaly Detection Server running on port ${PORT}`);
  console.log(`🔗 Frontend URL: ${process.env.FRONTEND_URL || 'http://localhost:3000'}`);
  console.log(`🤖 ML Model URL: ${process.env.ML_MODEL_URL || 'http://localhost:8000'}`);
});

export default app;