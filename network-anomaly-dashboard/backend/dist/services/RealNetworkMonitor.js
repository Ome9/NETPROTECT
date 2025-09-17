"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RealNetworkMonitor = void 0;
const child_process_1 = require("child_process");
const util_1 = require("util");
const execAsync = (0, util_1.promisify)(child_process_1.exec);
class RealNetworkMonitor {
    constructor() {
        this.isMonitoring = false;
        this.monitoringInterval = null;
        this.previousNetworkStats = new Map();
    }
    async startRealTimeMonitoring(intervalMs = 2000) {
        if (this.isMonitoring) {
            console.log('Real-time monitoring already running');
            return;
        }
        this.isMonitoring = true;
        console.log('🔄 Starting real Windows network monitoring...');
        // Initial data collection
        await this.collectInitialNetworkStats();
        this.monitoringInterval = setInterval(async () => {
            try {
                const metrics = await this.getRealSystemMetrics();
                // Emit to dashboard via socket or API
                this.onMetricsCollected(metrics);
            }
            catch (error) {
                console.error('Error collecting real metrics:', error);
            }
        }, intervalMs);
    }
    stopMonitoring() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }
        this.isMonitoring = false;
        console.log('✅ Real-time network monitoring stopped');
    }
    async getRealSystemMetrics() {
        const timestamp = new Date().toISOString();
        const [cpuUsage, memoryUsage, diskUsage, networkInterfaces, activeConnections, networkStats] = await Promise.all([
            this.getRealCpuUsage(),
            this.getRealMemoryUsage(),
            this.getRealDiskUsage(),
            this.getRealNetworkInterfaceStats(),
            this.getRealActiveConnections(),
            this.getRealNetworkStats()
        ]);
        return {
            timestamp,
            cpuUsage,
            memoryUsage,
            diskUsage,
            networkInterfaces,
            activeConnections,
            networkStats
        };
    }
    async getRealCpuUsage() {
        try {
            // Use PowerShell to get CPU usage
            const { stdout } = await execAsync(`
        powershell "Get-WmiObject -class win32_processor | 
        Measure-Object -property LoadPercentage -Average | 
        Select Average"
      `);
            const match = stdout.match(/Average\s*:\s*(\d+\.?\d*)/);
            return match ? parseFloat(match[1]) : 0;
        }
        catch (error) {
            console.error('Error getting CPU usage:', error);
            return 0;
        }
    }
    async getRealMemoryUsage() {
        try {
            const { stdout } = await execAsync(`
        powershell "
        $totalMemory = (Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory;
        $availableMemory = (Get-WmiObject -Class Win32_OperatingSystem).AvailablePhysicalMemory * 1024;
        $usedMemory = $totalMemory - $availableMemory;
        ($usedMemory / $totalMemory) * 100
        "
      `);
            return parseFloat(stdout.trim()) || 0;
        }
        catch (error) {
            console.error('Error getting memory usage:', error);
            return 0;
        }
    }
    async getRealDiskUsage() {
        try {
            const { stdout } = await execAsync(`
        powershell "Get-WmiObject -Class Win32_LogicalDisk -Filter \\"DriveType=3\\" | 
        ForEach-Object { [math]::Round(($_.Size - $_.FreeSpace) / $_.Size * 100, 2) } | 
        Measure-Object -Average | Select-Object Average"
      `);
            const match = stdout.match(/Average\s*:\s*(\d+\.?\d*)/);
            return match ? parseFloat(match[1]) : 0;
        }
        catch (error) {
            console.error('Error getting disk usage:', error);
            return 0;
        }
    }
    async getRealNetworkInterfaceStats() {
        try {
            const { stdout } = await execAsync(`
        powershell "
        Get-WmiObject -Class Win32_PerfRawData_Tcpip_NetworkInterface | 
        Where-Object { $_.Name -ne 'Loopback*' -and $_.Name -ne '_Total' -and $_.Name -ne 'MS TCP Loopback interface' } | 
        Select-Object Name, BytesReceivedPerSec, BytesSentPerSec, PacketsReceivedPerSec, PacketsSentPerSec | 
        ConvertTo-Json
        "
      `);
            const interfaces = JSON.parse(stdout || '[]');
            return Array.isArray(interfaces) ? interfaces : [interfaces];
        }
        catch (error) {
            console.error('Error getting network interface stats:', error);
            return [];
        }
    }
    async getRealActiveConnections() {
        try {
            // Get active connections with process info
            const { stdout } = await execAsync('netstat -ano');
            const connections = [];
            const lines = stdout.split('\n');
            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed || trimmed.includes('Proto') || trimmed.includes('Active'))
                    continue;
                const parts = trimmed.split(/\s+/);
                if (parts.length >= 4) {
                    const [protocol, localAddr, remoteAddr, state, ...pidParts] = parts;
                    const pid = pidParts.length > 0 ? parseInt(pidParts[0]) : undefined;
                    const [localAddress, localPort] = localAddr.split(':').length >= 2
                        ? [localAddr.substring(0, localAddr.lastIndexOf(':')), localAddr.substring(localAddr.lastIndexOf(':') + 1)]
                        : [localAddr, ''];
                    const [remoteAddress, remotePort] = remoteAddr.includes(':')
                        ? [remoteAddr.substring(0, remoteAddr.lastIndexOf(':')), remoteAddr.substring(remoteAddr.lastIndexOf(':') + 1)]
                        : [remoteAddr, ''];
                    connections.push({
                        protocol: protocol.toLowerCase(),
                        localAddress,
                        localPort,
                        remoteAddress,
                        remotePort,
                        state: state || 'UNKNOWN',
                        pid
                    });
                }
            }
            return connections;
        }
        catch (error) {
            console.error('Error getting active connections:', error);
            return [];
        }
    }
    async getRealNetworkStats() {
        try {
            const connections = await this.getRealActiveConnections();
            return {
                totalConnections: connections.length,
                tcpConnections: connections.filter(c => c.protocol === 'tcp').length,
                udpConnections: connections.filter(c => c.protocol === 'udp').length,
                listeningPorts: connections.filter(c => c.state === 'LISTENING').length
            };
        }
        catch (error) {
            console.error('Error calculating network stats:', error);
            return {
                totalConnections: 0,
                tcpConnections: 0,
                udpConnections: 0,
                listeningPorts: 0
            };
        }
    }
    async collectInitialNetworkStats() {
        try {
            const interfaces = await this.getRealNetworkInterfaceStats();
            interfaces.forEach(intf => {
                this.previousNetworkStats.set(intf.Name, {
                    bytesReceived: parseInt(intf.BytesReceivedPerSec) || 0,
                    bytesSent: parseInt(intf.BytesSentPerSec) || 0,
                    packetsReceived: parseInt(intf.PacketsReceivedPerSec) || 0,
                    packetsSent: parseInt(intf.PacketsSentPerSec) || 0
                });
            });
            console.log('📊 Collected initial network statistics for', interfaces.length, 'interfaces');
        }
        catch (error) {
            console.error('Error collecting initial network stats:', error);
        }
    }
    onMetricsCollected(metrics) {
        // This would emit to the dashboard via Socket.IO or API
        console.log(`📈 Real Metrics Collected:
    CPU: ${metrics.cpuUsage.toFixed(1)}%
    Memory: ${metrics.memoryUsage.toFixed(1)}%
    Disk: ${metrics.diskUsage.toFixed(1)}%
    Active Connections: ${metrics.activeConnections.length}
    Network Interfaces: ${metrics.networkInterfaces.length}
    `);
    }
    // Method to get process name by PID
    async getProcessNameByPid(pid) {
        try {
            const { stdout } = await execAsync(`
        powershell "Get-Process -Id ${pid} | Select-Object ProcessName"
      `);
            const match = stdout.match(/ProcessName\s*:\s*(.+)/);
            return match ? match[1].trim() : 'Unknown';
        }
        catch (error) {
            return 'Unknown';
        }
    }
    // Get detailed network traffic per process
    async getNetworkTrafficByProcess() {
        try {
            const { stdout } = await execAsync(`
        powershell "
        Get-NetTCPConnection | Group-Object OwningProcess | 
        ForEach-Object { 
          $process = Get-Process -Id $_.Name -ErrorAction SilentlyContinue;
          [PSCustomObject]@{
            ProcessName = if($process) { $process.ProcessName } else { 'Unknown' };
            PID = $_.Name;
            ConnectionCount = $_.Count;
            Connections = $_.Group | Select-Object LocalAddress, LocalPort, RemoteAddress, RemotePort, State
          }
        } | ConvertTo-Json -Depth 3
        "
      `);
            const processTraffic = JSON.parse(stdout || '[]');
            return Array.isArray(processTraffic) ? processTraffic : [processTraffic];
        }
        catch (error) {
            console.error('Error getting network traffic by process:', error);
            return [];
        }
    }
}
exports.RealNetworkMonitor = RealNetworkMonitor;
//# sourceMappingURL=RealNetworkMonitor.js.map