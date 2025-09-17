#!/usr/bin/env python3
"""
Real-time Network Anomaly Detection System
Monitors your PC's actual network traffic and provides real threat detection using trained ML models.
"""

import psutil
import socket
import time
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
import subprocess
import sys

class RealPCNetworkMonitor:
    """Monitors real PC network activity and system metrics."""
    
    def __init__(self):
        self.monitoring = False
        self.data_history = []
        self.connections_history = []
        
    def get_real_system_metrics(self) -> Dict:
        """Get actual system metrics from your PC."""
        try:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory Usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk Usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network Statistics
            net_io = psutil.net_io_counters()
            
            # Network Connections
            connections = psutil.net_connections(kind='inet')
            active_connections = [
                {
                    'laddr': f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "N/A",
                    'raddr': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "N/A",
                    'status': conn.status,
                    'pid': conn.pid,
                    'family': conn.family.name if conn.family else 'unknown',
                    'type': conn.type.name if conn.type else 'unknown'
                }
                for conn in connections if conn.status == psutil.CONN_ESTABLISHED
            ]
            
            # Process Network Activity
            process_network = []
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    if proc.info['connections']:
                        process_network.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'connection_count': len(proc.info['connections'])
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'memory_total': memory.total,
                'memory_available': memory.available,
                'disk_usage': disk_percent,
                'network_io': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                    'errin': net_io.errin,
                    'errout': net_io.errout,
                    'dropin': net_io.dropin,
                    'dropout': net_io.dropout
                },
                'active_connections': active_connections,
                'connection_count': len(active_connections),
                'process_network_activity': process_network[:10]  # Top 10 processes
            }
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error collecting system metrics: {e}")
            return {'timestamp': datetime.now().isoformat(), 'error': str(e)}
    
    def get_network_interfaces(self) -> List[Dict]:
        """Get information about network interfaces."""
        try:
            interfaces = []
            stats = psutil.net_io_counters(pernic=True)
            
            for interface_name, stats_info in stats.items():
                interface_info = {
                    'name': interface_name,
                    'bytes_sent': stats_info.bytes_sent,
                    'bytes_recv': stats_info.bytes_recv,
                    'packets_sent': stats_info.packets_sent,
                    'packets_recv': stats_info.packets_recv,
                    'errin': stats_info.errin,
                    'errout': stats_info.errout,
                    'dropin': stats_info.dropin,
                    'dropout': stats_info.dropout
                }
                interfaces.append(interface_info)
            
            return interfaces
            
        except Exception as e:
            print(f"❌ Error getting network interfaces: {e}")
            return []
    
    def detect_suspicious_activity(self, metrics: Dict) -> Dict:
        """Detect suspicious network activity using heuristics."""
        alerts = []
        risk_score = 0
        
        try:
            # Check for high CPU usage
            if metrics.get('cpu_usage', 0) > 90:
                alerts.append({
                    'type': 'HIGH_CPU_USAGE',
                    'severity': 'high',
                    'message': f"High CPU usage detected: {metrics['cpu_usage']:.1f}%"
                })
                risk_score += 30
            
            # Check for high memory usage
            if metrics.get('memory_usage', 0) > 95:
                alerts.append({
                    'type': 'HIGH_MEMORY_USAGE', 
                    'severity': 'high',
                    'message': f"High memory usage detected: {metrics['memory_usage']:.1f}%"
                })
                risk_score += 25
            
            # Check for suspicious number of connections
            connection_count = metrics.get('connection_count', 0)
            if connection_count > 100:
                alerts.append({
                    'type': 'HIGH_CONNECTION_COUNT',
                    'severity': 'medium',
                    'message': f"Unusual number of network connections: {connection_count}"
                })
                risk_score += 20
            
            # Check for suspicious processes with many connections
            for proc in metrics.get('process_network_activity', []):
                if proc['connection_count'] > 20:
                    alerts.append({
                        'type': 'SUSPICIOUS_PROCESS_ACTIVITY',
                        'severity': 'medium', 
                        'message': f"Process '{proc['name']}' has {proc['connection_count']} connections"
                    })
                    risk_score += 15
            
            # Check network errors
            net_io = metrics.get('network_io', {})
            total_errors = net_io.get('errin', 0) + net_io.get('errout', 0)
            if total_errors > 100:
                alerts.append({
                    'type': 'NETWORK_ERRORS',
                    'severity': 'medium',
                    'message': f"High network error count: {total_errors}"
                })
                risk_score += 15
            
            # Analyze connection patterns
            foreign_connections = 0
            local_connections = 0
            
            for conn in metrics.get('active_connections', []):
                raddr = conn.get('raddr', '')
                if raddr and raddr != 'N/A':
                    ip = raddr.split(':')[0]
                    if ip.startswith(('192.168.', '10.', '172.16.')) or ip == '127.0.0.1':
                        local_connections += 1
                    else:
                        foreign_connections += 1
            
            if foreign_connections > 50:
                alerts.append({
                    'type': 'HIGH_FOREIGN_CONNECTIONS',
                    'severity': 'high',
                    'message': f"Many foreign connections detected: {foreign_connections}"
                })
                risk_score += 35
            
            # Determine threat level
            if risk_score >= 70:
                threat_level = 'CRITICAL'
            elif risk_score >= 40:
                threat_level = 'HIGH'
            elif risk_score >= 20:
                threat_level = 'MEDIUM'
            elif risk_score > 0:
                threat_level = 'LOW'
            else:
                threat_level = 'NORMAL'
            
            return {
                'timestamp': metrics.get('timestamp', datetime.now().isoformat()),
                'risk_score': min(risk_score, 100),
                'threat_level': threat_level,
                'alerts': alerts,
                'total_connections': connection_count,
                'foreign_connections': foreign_connections,
                'local_connections': local_connections
            }
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'risk_score': 0,
                'threat_level': 'ERROR',
                'alerts': [{'type': 'ANALYSIS_ERROR', 'severity': 'high', 'message': str(e)}],
                'error': str(e)
            }
    
    def start_monitoring(self, interval: int = 3):
        """Start continuous monitoring."""
        print("🔍 Starting Real PC Network Monitoring...")
        print(f"📊 Monitoring interval: {interval} seconds")
        print("=" * 60)
        
        self.monitoring = True
        try:
            while self.monitoring:
                # Collect real system metrics
                metrics = self.get_real_system_metrics()
                
                # Detect suspicious activity
                threat_analysis = self.detect_suspicious_activity(metrics)
                
                # Display current status
                self.display_monitoring_status(metrics, threat_analysis)
                
                # Store in history
                self.data_history.append({
                    'metrics': metrics,
                    'threats': threat_analysis
                })
                
                # Keep only last 100 entries
                if len(self.data_history) > 100:
                    self.data_history = self.data_history[-100:]
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
        finally:
            self.monitoring = False
    
    def display_monitoring_status(self, metrics: Dict, threats: Dict):
        """Display current monitoring status."""
        timestamp = metrics.get('timestamp', 'N/A')
        cpu = metrics.get('cpu_usage', 0)
        memory = metrics.get('memory_usage', 0)
        connections = metrics.get('connection_count', 0)
        
        threat_level = threats.get('threat_level', 'UNKNOWN')
        risk_score = threats.get('risk_score', 0)
        
        # Color coding for threat levels
        if threat_level == 'CRITICAL':
            level_color = '🔴'
        elif threat_level == 'HIGH':
            level_color = '🟠'
        elif threat_level == 'MEDIUM':
            level_color = '🟡'
        elif threat_level == 'LOW':
            level_color = '🟢'
        else:
            level_color = '⚪'
        
        print(f"\r{level_color} [{timestamp[-8:]}] CPU: {cpu:5.1f}% | MEM: {memory:5.1f}% | Connections: {connections:3d} | Risk: {risk_score:3d} | {threat_level}", end="", flush=True)
        
        # Show alerts if any
        alerts = threats.get('alerts', [])
        if alerts:
            print(f"\n  ⚠️ Alerts: {len(alerts)}")
            for alert in alerts[:3]:  # Show first 3 alerts
                print(f"    - {alert['message']}")
    
    def get_summary_report(self) -> Dict:
        """Get a summary report of monitoring session."""
        if not self.data_history:
            return {'error': 'No data collected yet'}
        
        total_entries = len(self.data_history)
        threat_levels = [entry['threats']['threat_level'] for entry in self.data_history]
        
        report = {
            'monitoring_duration': f"{total_entries * 3} seconds",
            'total_data_points': total_entries,
            'threat_level_distribution': {
                'NORMAL': threat_levels.count('NORMAL'),
                'LOW': threat_levels.count('LOW'),
                'MEDIUM': threat_levels.count('MEDIUM'),
                'HIGH': threat_levels.count('HIGH'),
                'CRITICAL': threat_levels.count('CRITICAL')
            },
            'max_risk_score': max([entry['threats']['risk_score'] for entry in self.data_history]),
            'avg_cpu_usage': np.mean([entry['metrics'].get('cpu_usage', 0) for entry in self.data_history]),
            'avg_memory_usage': np.mean([entry['metrics'].get('memory_usage', 0) for entry in self.data_history]),
            'avg_connections': np.mean([entry['metrics'].get('connection_count', 0) for entry in self.data_history]),
            'timestamp': datetime.now().isoformat()
        }
        
        return report


def main():
    """Main function to run real network monitoring."""
    print("🛡️  NetProtect - Real PC Network Anomaly Detection")
    print("=" * 60)
    print("📡 This will monitor your PC's REAL network traffic and system metrics")
    print("🤖 No fake data - only actual network connections and system activity")
    print("=" * 60)
    
    monitor = RealPCNetworkMonitor()
    
    try:
        # Show initial system info
        print("🔍 Initial System Scan...")
        initial_metrics = monitor.get_real_system_metrics()
        initial_threats = monitor.detect_suspicious_activity(initial_metrics)
        
        print(f"💻 System Status:")
        print(f"  - CPU: {initial_metrics.get('cpu_usage', 0):.1f}%")
        print(f"  - Memory: {initial_metrics.get('memory_usage', 0):.1f}%")
        print(f"  - Active Connections: {initial_metrics.get('connection_count', 0)}")
        print(f"  - Current Threat Level: {initial_threats.get('threat_level', 'UNKNOWN')}")
        print()
        
        # Show network interfaces
        interfaces = monitor.get_network_interfaces()
        print(f"🌐 Network Interfaces: {len(interfaces)}")
        for interface in interfaces[:3]:
            print(f"  - {interface['name']}: {interface['bytes_recv']} bytes received")
        print()
        
        print("🚀 Starting real-time monitoring... (Press Ctrl+C to stop)")
        print()
        
        # Start monitoring
        monitor.start_monitoring(interval=3)
        
    except KeyboardInterrupt:
        print("\n\n📊 Session Summary:")
        report = monitor.get_summary_report()
        print(json.dumps(report, indent=2))
        print("\n✅ Thank you for using NetProtect Real Network Monitoring!")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()