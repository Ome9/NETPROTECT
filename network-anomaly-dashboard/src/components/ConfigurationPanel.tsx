import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { X, Settings, Save, RefreshCw } from 'lucide-react';
import { Socket } from 'socket.io-client';

interface ConfigurationPanelProps {
  socket: Socket | null;
  onClose: () => void;
}

export const ConfigurationPanel: React.FC<ConfigurationPanelProps> = ({ socket, onClose }) => {
  const [config, setConfig] = useState({
    mlModelUrl: 'http://localhost:8000',
    logFilePath: '',
    refreshInterval: 1000,
    maxHistoryPoints: 100,
    enableRealTimeAlerts: true,
    confidenceThreshold: 0.7
  });
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    if (!socket) return;
    
    setIsSaving(true);
    try {
      // Send configuration to backend
      socket.emit('configure-monitoring', config);
      
      // Show success message
      setTimeout(() => {
        setIsSaving(false);
        onClose();
      }, 1000);
    } catch (error) {
      console.error('Failed to save configuration:', error);
      setIsSaving(false);
    }
  };

  const handleInputChange = (field: string, value: string | number | boolean) => {
    setConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  return (
    <Card className="bg-gray-800/95 border-gray-700 backdrop-blur-sm">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <div className="flex items-center space-x-2">
          <Settings className="w-5 h-5 text-blue-400" />
          <CardTitle className="text-gray-200">Dashboard Configuration</CardTitle>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={onClose}
          className="border-gray-600 hover:border-gray-400"
        >
          <X className="w-4 h-4" />
        </Button>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* ML Model Configuration */}
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-gray-200">ML Model Settings</h3>
          
          <div className="space-y-2">
            <label className="text-xs text-gray-400">Model API URL</label>
            <Input
              type="text"
              value={config.mlModelUrl}
              onChange={(e) => handleInputChange('mlModelUrl', e.target.value)}
              placeholder="http://localhost:8000"
              className="bg-gray-900 border-gray-600 text-gray-200"
            />
          </div>
          
          <div className="space-y-2">
            <label className="text-xs text-gray-400">Confidence Threshold</label>
            <Input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={config.confidenceThreshold}
              onChange={(e) => handleInputChange('confidenceThreshold', parseFloat(e.target.value))}
              className="bg-gray-900 border-gray-600 text-gray-200"
            />
            <p className="text-xs text-gray-500">
              Minimum confidence level to trigger anomaly alerts (0.0 - 1.0)
            </p>
          </div>
        </div>

        {/* Network Monitoring */}
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-gray-200">Network Monitoring</h3>
          
          <div className="space-y-2">
            <label className="text-xs text-gray-400">Log File Path (Optional)</label>
            <Input
              type="text"
              value={config.logFilePath}
              onChange={(e) => handleInputChange('logFilePath', e.target.value)}
              placeholder="C:\\logs\\network.log"
              className="bg-gray-900 border-gray-600 text-gray-200"
            />
            <p className="text-xs text-gray-500">
              Path to network log file for real-time monitoring
            </p>
          </div>
          
          <div className="space-y-2">
            <label className="text-xs text-gray-400">Refresh Interval (ms)</label>
            <Input
              type="number"
              min="500"
              max="10000"
              step="100"
              value={config.refreshInterval}
              onChange={(e) => handleInputChange('refreshInterval', parseInt(e.target.value))}
              className="bg-gray-900 border-gray-600 text-gray-200"
            />
          </div>
        </div>

        {/* Display Settings */}
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-gray-200">Display Settings</h3>
          
          <div className="space-y-2">
            <label className="text-xs text-gray-400">Max History Points</label>
            <Input
              type="number"
              min="50"
              max="500"
              step="10"
              value={config.maxHistoryPoints}
              onChange={(e) => handleInputChange('maxHistoryPoints', parseInt(e.target.value))}
              className="bg-gray-900 border-gray-600 text-gray-200"
            />
            <p className="text-xs text-gray-500">
              Maximum number of data points to display in charts
            </p>
          </div>
        </div>

        {/* Current Status */}
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-gray-200">Connection Status</h3>
          
          <div className="flex flex-wrap gap-2">
            <Badge variant={socket?.connected ? "default" : "destructive"}>
              WebSocket: {socket?.connected ? 'Connected' : 'Disconnected'}
            </Badge>
            <Badge variant="outline">
              Backend: Ready
            </Badge>
            <Badge variant="outline">
              ML Model: Checking...
            </Badge>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-between pt-4 border-t border-gray-700">
          <Button
            variant="outline"
            onClick={() => window.location.reload()}
            className="border-gray-600 hover:border-gray-400"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Reset Dashboard
          </Button>
          
          <Button
            onClick={handleSave}
            disabled={isSaving}
            className="bg-blue-600 hover:bg-blue-700"
          >
            {isSaving ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                Save Configuration
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};