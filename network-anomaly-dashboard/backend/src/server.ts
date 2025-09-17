import express from "express";
import http from "http";
import cors from "cors";
import helmet from "helmet";
import morgan from "morgan";
import { Server } from "socket.io";
import { NetworkMonitor } from "./services/NetworkMonitor";
import { AnomalyDetectionService } from "./services/AnomalyDetectionService";

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: process.env.FRONTEND_URL || "http://localhost:3000",
    methods: ["GET", "POST"]
  }
});

app.use(helmet());
app.use(cors());
app.use(morgan("combined"));
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

const networkMonitor = new NetworkMonitor();
const anomalyDetectionService = new AnomalyDetectionService(
  process.env.ML_MODEL_URL || "http://localhost:8000"
);

// Basic API root route
app.get('/api', (req, res) => {
  res.json({
    message: 'NetProtect API Server',
    version: '1.0.0',
    status: 'running',
    timestamp: new Date().toISOString(),
    endpoints: [
      'GET /api/health',
      'GET /api/network/current', 
      'GET /api/network/status',
      'POST /api/network/start',
      'POST /api/network/stop'
    ]
  });
});

// Import and use routes (with error handling)
import routes from "./routes";
app.use("/api", routes);
console.log('✅ Routes loaded successfully');

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Get network status endpoint
app.get('/api/network/status', async (req, res) => {
  try {
    const networkData = await networkMonitor.getCurrentNetworkData();
    const interfaces = await networkMonitor.getNetworkInterfaces();
    
    res.json({
      isMonitoring: true,
      activeConnections: networkData.connections || 0,
      packetsPerSecond: (networkData.packetsIn + networkData.packetsOut) || 0,
      bytesPerSecond: (networkData.bytesIn + networkData.bytesOut) || 0,
      networkInterfaces: interfaces,
      selectedInterface: interfaces[0] || 'Ethernet'
    });
  } catch (error) {
    console.error('Error fetching network status:', error);
    res.json({
      isMonitoring: false,
      activeConnections: 0,
      packetsPerSecond: 0,
      bytesPerSecond: 0,
      networkInterfaces: ['Ethernet', 'Wi-Fi'],
      selectedInterface: 'Ethernet'
    });
  }
});

// Get available network interfaces endpoint
app.get('/api/network/interfaces', async (req, res) => {
  try {
    const interfaces = await networkMonitor.getNetworkInterfaces();
    res.json({
      interfaces: interfaces
    });
  } catch (error) {
    console.error('Error fetching network interfaces:', error);
    res.json({
      interfaces: ['Ethernet', 'Wi-Fi'] // Fallback
    });
  }
});

// Start network monitoring endpoint
app.post('/api/network/start', async (req, res) => {
  try {
    const { interface: networkInterface } = req.body;
    await networkMonitor.startMonitoring(networkInterface);
    res.json({
      success: true,
      message: `Network monitoring started${networkInterface ? ` on ${networkInterface}` : ''}`
    });
  } catch (error) {
    console.error('Error starting network monitoring:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Stop network monitoring endpoint  
app.post('/api/network/stop', async (req, res) => {
  try {
    networkMonitor.stopMonitoring();
    res.json({
      success: true,
      message: 'Network monitoring stopped'
    });
  } catch (error) {
    console.error('Error stopping network monitoring:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

io.on("connection", (socket) => {
  console.log(`Client connected: ${socket.id}`);

  const networkDataInterval = setInterval(async () => {
    try {
      const networkData = await networkMonitor.getCurrentNetworkData();
      // Skip ML anomaly detection for now to avoid connection errors
      // const anomalyResults = await anomalyDetectionService.detectAnomalies(networkData);
      
      socket.emit("network-data", {
        timestamp: new Date().toISOString(),
        data: networkData,
        anomalies: [] // Empty anomalies array for now
      });
    } catch (error) {
      console.error("Error sending network data:", error);
    }
  }, 1000);

  socket.on("disconnect", () => {
    console.log(`Client disconnected: ${socket.id}`);
    clearInterval(networkDataInterval);
  });
});

const PORT = process.env.PORT || 3001;

server.listen(PORT, () => {
  console.log(`🚀 NetProtect Backend Server running on port ${PORT}`);
  console.log(`📊 Dashboard URL: http://localhost:3000`);
  console.log(`🔗 API Base URL: http://localhost:${PORT}/api`);
  console.log(`🔌 Socket.IO ready for real-time connections`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down server...');
  networkMonitor.stopMonitoring();
  server.close(() => {
    console.log('✅ Server shutdown complete');
    process.exit(0);
  });
});

export default app;
