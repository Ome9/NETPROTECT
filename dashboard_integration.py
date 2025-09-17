#!/usr/bin/env python3
"""
Dashboard Integration Script
Runs real network monitoring and provides data to the React dashboard via WebSocket
"""

import asyncio
import websockets
import json
import threading
import time
from real_network_monitor import RealPCNetworkMonitor
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardIntegration:
    """Integrates real network monitoring with the React dashboard."""
    
    def __init__(self, websocket_port=8765):
        self.websocket_port = websocket_port
        self.monitor = RealPCNetworkMonitor()
        self.connected_clients = set()
        self.current_data = None
        self.monitoring_thread = None
        
    def start_monitoring_thread(self):
        """Start monitoring in a separate thread."""
        def monitor_loop():
            logger.info("🔍 Starting background network monitoring...")
            while True:
                try:
                    # Get real system metrics
                    metrics = self.monitor.get_real_system_metrics()
                    
                    # Detect threats
                    threats = self.monitor.detect_suspicious_activity(metrics)
                    
                    # Combine data
                    self.current_data = {
                        'timestamp': datetime.now().isoformat(),
                        'type': 'real-network-data',
                        'systemMetrics': metrics,
                        'threatAnalysis': threats,
                        'dataSource': 'REAL',
                        'status': 'active'
                    }
                    
                    # Send to all connected clients
                    if self.connected_clients:
                        asyncio.create_task(self.broadcast_data())
                    
                    time.sleep(3)  # Update every 3 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Monitoring error: {e}")
                    time.sleep(5)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    async def broadcast_data(self):
        """Broadcast current data to all connected clients."""
        if not self.current_data or not self.connected_clients:
            return
        
        message = json.dumps(self.current_data)
        disconnected_clients = set()
        
        for client in self.connected_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending data to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
    
    async def handle_client(self, websocket, path):
        """Handle incoming WebSocket connections."""
        logger.info(f"🔌 Client connected from {websocket.remote_address}")
        self.connected_clients.add(websocket)
        
        try:
            # Send welcome message
            welcome = {
                'type': 'connection-established',
                'message': 'Connected to NetProtect Real Network Monitor',
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(welcome))
            
            # Send current data if available
            if self.current_data:
                await websocket.send(json.dumps(self.current_data))
            
            # Keep connection alive and handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 Client disconnected")
        except Exception as e:
            logger.error(f"Client handling error: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def handle_client_message(self, websocket, data):
        """Handle messages from clients."""
        message_type = data.get('type', '')
        
        if message_type == 'request-current-data':
            if self.current_data:
                await websocket.send(json.dumps(self.current_data))
        
        elif message_type == 'request-summary':
            summary = self.monitor.get_summary_report()
            response = {
                'type': 'summary-report',
                'data': summary,
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
        
        elif message_type == 'ping':
            pong = {
                'type': 'pong',
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(pong))
        
        else:
            logger.info(f"Unknown message type: {message_type}")
    
    async def start_websocket_server(self):
        """Start the WebSocket server."""
        logger.info(f"🌐 Starting WebSocket server on port {self.websocket_port}")
        
        server = await websockets.serve(
            self.handle_client,
            "localhost", 
            self.websocket_port
        )
        
        logger.info(f"✅ WebSocket server ready at ws://localhost:{self.websocket_port}")
        logger.info("📊 Dashboard can now connect to receive real network data")
        
        await server.wait_closed()
    
    def run(self):
        """Run the integration service."""
        print("🛡️  NetProtect - Dashboard Integration Service")
        print("=" * 60)
        print("🔄 This service provides REAL network data to your dashboard")
        print("📡 Dashboard will show actual PC network traffic and threats")
        print("=" * 60)
        
        # Start monitoring thread
        self.start_monitoring_thread()
        
        # Start WebSocket server
        try:
            asyncio.run(self.start_websocket_server())
        except KeyboardInterrupt:
            print("\n🛑 Integration service stopped")
        except Exception as e:
            print(f"❌ Service error: {e}")


def main():
    """Main function."""
    integration = DashboardIntegration()
    integration.run()


if __name__ == "__main__":
    main()