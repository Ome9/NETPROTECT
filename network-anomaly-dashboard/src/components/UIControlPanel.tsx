'use client';

import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { EnhancedSlider } from '@/components/ui/enhanced-slider';
import { 
  Settings, Palette, Monitor, Activity,
  BarChart3, Network, Shield, Zap, RefreshCw, 
  SunMoon, Volume2, VolumeX, Bell, BellOff,
  Grid, Layout, Maximize2, Minimize2
} from 'lucide-react';

// Inline Switch component to avoid import issues
const Switch: React.FC<{
  checked?: boolean;
  onCheckedChange?: (checked: boolean) => void;
  disabled?: boolean;
  className?: string;
}> = ({ checked = false, onCheckedChange, disabled = false, className = '' }) => {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      className={`
        peer inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full 
        border-2 border-transparent transition-colors 
        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 
        focus-visible:ring-offset-2 focus-visible:ring-offset-black 
        disabled:cursor-not-allowed disabled:opacity-50 
        ${checked 
          ? 'bg-gradient-to-r from-blue-500 to-purple-600' 
          : 'bg-gray-700'
        } 
        ${className}
      `}
      onClick={() => !disabled && onCheckedChange?.(!checked)}
    >
      <div
        className={`
          pointer-events-none block h-5 w-5 rounded-full bg-white shadow-lg 
          ring-0 transition-transform 
          ${checked ? 'translate-x-5' : 'translate-x-0'}
        `}
      />
    </button>
  );
};

interface UIControlPanelProps {
  onThemeChange?: (theme: string) => void;
  onLayoutChange?: (layout: string) => void;
  onSectionToggle?: (sectionId: string, enabled: boolean) => void;
  onClose?: () => void;
  currentSettings?: {
    darkMode: boolean;
    compactMode: boolean;
    animations: boolean;
    transparency: number;
    glowEffects: boolean;
    sidebarCollapsed: boolean;
    fullscreenMode: boolean;
    gridDensity: 'compact' | 'normal' | 'spacious';
    colorTheme: string;
    soundEnabled: boolean;
    notifications: boolean;
    alertLevel: 'low' | 'medium' | 'high';
    refreshRate: number;
    dataRetention: number;
    realtimeUpdates: boolean;
  };
  visibleSections?: {
    systemMetrics: boolean;
    networkTopology: boolean;
    threatDetection: boolean;
    trafficAnalysis: boolean;
    modelMonitoring: boolean;
    configuration: boolean;
  };
  onSettingChange?: (key: string, value: boolean | string | number) => void;
}

export const UIControlPanel: React.FC<UIControlPanelProps> = ({
  onThemeChange,
  onLayoutChange,
  onSectionToggle,
  onClose,
  currentSettings,
  visibleSections,
  onSettingChange
}) => {
  const [activeSection, setActiveSection] = useState<string>('display');

  // Use provided sections or defaults
  const sections = visibleSections || {
    systemMetrics: true,
    networkTopology: true,
    threatDetection: true,
    trafficAnalysis: true,
    modelMonitoring: true,
    configuration: true
  };

  const updateSetting = useCallback((key: string, value: boolean | string | number) => {
    console.log('🎛️ UI Control Update:', { key, value, currentSettings });
    
    // Call the parent's setting change handler
    onSettingChange?.(key, value);
    
    // Trigger specific callbacks for external components
    if (key === 'darkMode') {
      onThemeChange?.(value ? 'dark' : 'light');
    }
    if (key === 'gridDensity' || key === 'sidebarCollapsed') {
      onLayoutChange?.(key);
    }
    if (key.endsWith('Metrics') || key.endsWith('Topology') || key.endsWith('Detection') || 
        key.endsWith('Analysis') || key.endsWith('Monitoring') || key.endsWith('configuration')) {
      onSectionToggle?.(key, value as boolean);
    }
  }, [onThemeChange, onLayoutChange, onSectionToggle, onSettingChange, currentSettings]);

  const controlSections = [
    {
      id: 'display',
      title: 'Display',
      icon: <Monitor className="h-4 w-4" />,
      description: 'Theme, colors, and visual effects'
    },
    {
      id: 'layout',
      title: 'Layout',
      icon: <Layout className="h-4 w-4" />,
      description: 'Dashboard layout and structure'
    },
    {
      id: 'sections',
      title: 'Sections',
      icon: <Grid className="h-4 w-4" />,
      description: 'Show/hide dashboard components'
    },
    {
      id: 'alerts',
      title: 'Alerts',
      icon: <Bell className="h-4 w-4" />,
      description: 'Notification and sound settings'
    },
    {
      id: 'performance',
      title: 'Performance',
      icon: <Activity className="h-4 w-4" />,
      description: 'Update rates and data settings'
    }
  ];

  const renderDisplayControls = () => (
    <div className="space-y-6">
      {/* Theme Controls */}
      <div className="space-y-4">
        <h4 className="text-white font-medium">Theme Settings</h4>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Dark Mode</span>
          <Switch 
            checked={currentSettings?.darkMode || false}
            onCheckedChange={(checked: boolean) => updateSetting('darkMode', checked)}
          />
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Glow Effects</span>
          <Switch 
            checked={currentSettings?.glowEffects || false}
            onCheckedChange={(checked: boolean) => updateSetting('glowEffects', checked)}
          />
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Animations</span>
          <Switch 
            checked={currentSettings?.animations || false}
            onCheckedChange={(checked: boolean) => updateSetting('animations', checked)}
          />
        </div>
      </div>

      {/* Visual Effects */}
      <div className="space-y-4">
        <h4 className="text-white font-medium">Visual Effects</h4>
        <EnhancedSlider
          label="Background Transparency"
          variant="gemini"
          value={currentSettings?.transparency || 85}
          onChange={(value) => updateSetting('transparency', value)}
          min={50}
          max={100}
          className="mb-4"
        />
      </div>

      {/* Color Themes */}
      <div className="space-y-4">
        <h4 className="text-white font-medium">Color Themes</h4>
        <div className="grid grid-cols-2 gap-3">
          {[
            { 
              name: 'Cyberpunk', 
              color: 'from-purple-500 via-pink-500 to-red-500',
              accent: '#ff00ff',
              description: 'Futuristic neon vibes'
            },
            { 
              name: 'Ocean Blue', 
              color: 'from-blue-600 via-cyan-500 to-teal-400',
              accent: '#0ea5e9',
              description: 'Deep ocean depths'
            },
            { 
              name: 'Forest Green', 
              color: 'from-emerald-600 via-green-500 to-lime-400',
              accent: '#10b981',
              description: 'Nature\'s serenity'
            },
            { 
              name: 'Sunset Orange', 
              color: 'from-orange-500 via-red-500 to-pink-500',
              accent: '#f97316',
              description: 'Warm twilight glow'
            },
            { 
              name: 'Purple Galaxy', 
              color: 'from-indigo-600 via-purple-600 to-pink-600',
              accent: '#8b5cf6',
              description: 'Cosmic wonder'
            },
            { 
              name: 'Arctic Ice', 
              color: 'from-slate-400 via-blue-300 to-cyan-200',
              accent: '#64748b',
              description: 'Cool crystalline'
            },
            { 
              name: 'Golden Hour', 
              color: 'from-amber-500 via-orange-400 to-yellow-300',
              accent: '#f59e0b',
              description: 'Warm golden light'
            },
            { 
              name: 'Midnight', 
              color: 'from-gray-800 via-slate-700 to-zinc-600',
              accent: '#374151',
              description: 'Dark elegance'
            }
          ].map(theme => (
            <div key={theme.name} className="relative group">
              <Button
                variant="ghost"
                className={`h-16 w-full bg-gradient-to-r ${theme.color} text-white hover:opacity-80 hover:scale-105 transition-all duration-300 flex flex-col items-center justify-center relative overflow-hidden ${
                  currentSettings?.colorTheme === theme.name.toLowerCase() ? 'ring-2 ring-white ring-opacity-50' : ''
                }`}
                onClick={() => updateSetting('colorTheme', theme.name.toLowerCase())}
              >
                <div className="text-sm font-bold">{theme.name}</div>
                <div className="text-xs opacity-80">{theme.description}</div>
                {currentSettings?.colorTheme === theme.name.toLowerCase() && (
                  <div className="absolute top-1 right-1 w-3 h-3 bg-white rounded-full"></div>
                )}
              </Button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderLayoutControls = () => (
    <div className="space-y-6">
      <div className="space-y-4">
        <h4 className="text-white font-medium">Layout Options</h4>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Compact Mode</span>
          <Switch 
            checked={currentSettings?.compactMode || false}
            onCheckedChange={(checked: boolean) => updateSetting('compactMode', checked)}
          />
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Sidebar Collapsed</span>
          <Switch 
            checked={currentSettings?.sidebarCollapsed || false}
            onCheckedChange={(checked: boolean) => updateSetting('sidebarCollapsed', checked)}
          />
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Fullscreen Mode</span>
          <Switch 
            checked={currentSettings?.fullscreenMode || false}
            onCheckedChange={(checked: boolean) => updateSetting('fullscreenMode', checked)}
          />
        </div>
      </div>

      <div className="space-y-4">
        <h4 className="text-white font-medium">Grid Density</h4>
        <div className="grid grid-cols-3 gap-2">
          {[
            { value: 'compact', label: 'Compact', icon: <Minimize2 className="h-4 w-4" /> },
            { value: 'normal', label: 'Normal', icon: <Grid className="h-4 w-4" /> },
            { value: 'spacious', label: 'Spacious', icon: <Maximize2 className="h-4 w-4" /> }
          ].map(option => (
            <Button
              key={option.value}
              variant={currentSettings?.gridDensity === option.value ? "rainbow" : "ghost"}
              className="flex flex-col items-center p-3 h-auto"
              onClick={() => updateSetting('gridDensity', option.value)}
            >
              {option.icon}
              <span className="text-xs mt-1">{option.label}</span>
            </Button>
          ))}
        </div>
      </div>
    </div>
  );

  const renderSectionControls = () => (
    <div className="space-y-6">
      <h4 className="text-white font-medium">Dashboard Sections</h4>
      {[
        { key: 'systemMetrics', label: 'System Metrics', icon: <BarChart3 className="h-4 w-4" /> },
        { key: 'networkTopology', label: 'Network Topology', icon: <Network className="h-4 w-4" /> },
        { key: 'threatDetection', label: 'Threat Detection', icon: <Shield className="h-4 w-4" /> },
        { key: 'trafficAnalysis', label: 'Traffic Analysis', icon: <Activity className="h-4 w-4" /> },
        { key: 'modelMonitoring', label: 'ML Model Monitoring', icon: <Zap className="h-4 w-4" /> },
        { key: 'configuration', label: 'Configuration Panel', icon: <Settings className="h-4 w-4" /> }
      ].map(section => (
        <div key={section.key} className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/10">
          <div className="flex items-center gap-3">
            {section.icon}
            <span className="text-gray-300">{section.label}</span>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={sections[section.key as keyof typeof sections] ? "neon" : "outline"}>
              {sections[section.key as keyof typeof sections] ? "Visible" : "Hidden"}
            </Badge>
            <Switch 
              checked={sections[section.key as keyof typeof sections] as boolean}
              onCheckedChange={(checked: boolean) => updateSetting(section.key, checked)}
            />
          </div>
        </div>
      ))}
    </div>
  );

  const renderAlertControls = () => (
    <div className="space-y-6">
      <div className="space-y-4">
        <h4 className="text-white font-medium">Notification Settings</h4>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Enable Notifications</span>
          <Switch 
            checked={currentSettings?.notifications || false}
            onCheckedChange={(checked: boolean) => updateSetting('notifications', checked)}
          />
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Sound Alerts</span>
          <Switch 
            checked={currentSettings?.soundEnabled || false}
            onCheckedChange={(checked: boolean) => updateSetting('soundEnabled', checked)}
          />
        </div>
      </div>

      <div className="space-y-4">
        <h4 className="text-white font-medium">Alert Level</h4>
        <div className="grid grid-cols-3 gap-2">
          {[
            { value: 'low', label: 'Low', color: 'text-green-400' },
            { value: 'medium', label: 'Medium', color: 'text-yellow-400' },
            { value: 'high', label: 'High', color: 'text-red-400' }
          ].map(level => (
            <Button
              key={level.value}
              variant={currentSettings?.alertLevel === level.value ? "neon" : "ghost"}
              className={`${level.color}`}
              onClick={() => updateSetting('alertLevel', level.value)}
            >
              {level.label}
            </Button>
          ))}
        </div>
      </div>
    </div>
  );

  const renderPerformanceControls = () => (
    <div className="space-y-6">
      <div className="space-y-4">
        <h4 className="text-white font-medium">Update Settings</h4>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Real-time Updates</span>
          <Switch 
            checked={currentSettings?.realtimeUpdates || false}
            onCheckedChange={(checked: boolean) => updateSetting('realtimeUpdates', checked)}
          />
        </div>
      </div>

      <div className="space-y-4">
        <EnhancedSlider
          label={`Refresh Rate: ${currentSettings?.refreshRate || 10}s`}
          variant="neon"
          glowColor="blue"
          value={currentSettings?.refreshRate || 10}
          onChange={(value) => updateSetting('refreshRate', value)}
          min={1}
          max={30}
        />
        
        <EnhancedSlider
          label={`Data Retention: ${currentSettings?.dataRetention || 24}h`}
          variant="gemini"
          value={currentSettings?.dataRetention || 24}
          onChange={(value) => updateSetting('dataRetention', value)}
          min={1}
          max={72}
        />
      </div>

      <Button 
        variant="rainbow" 
        className="w-full"
        onClick={() => {
          // Reset to default values
          onSettingChange?.('refreshRate', 5);
          onSettingChange?.('dataRetention', 24);
          onSettingChange?.('realtimeUpdates', true);
        }}
      >
        <RefreshCw className="h-4 w-4 mr-2" />
        Reset to Defaults
      </Button>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center space-y-4 relative">
        {onClose && (
          <button
            onClick={onClose}
            className="absolute top-0 right-0 text-white/70 hover:text-white text-2xl font-bold w-8 h-8 flex items-center justify-center rounded-full hover:bg-white/10 transition-colors"
          >
            ×
          </button>
        )}
        <motion.h1 
          className="text-4xl font-bold text-white"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          UI Control Center
        </motion.h1>
        <p className="text-gray-300 text-lg">
          Customize and control all dashboard components from one central location
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Control Section Tabs */}
        <div className="lg:col-span-1">
          <Card variant="gemini">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Control Sections
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {controlSections.map((section) => (
                <Button
                  key={section.id}
                  variant={activeSection === section.id ? "rainbow" : "ghost"}
                  className="w-full justify-start p-3 h-auto"
                  onClick={() => setActiveSection(section.id)}
                >
                  <div className="flex items-center gap-3">
                    {section.icon}
                    <div className="text-left">
                      <div className="font-medium">{section.title}</div>
                      <div className="text-xs text-gray-400">{section.description}</div>
                    </div>
                  </div>
                </Button>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Control Panel */}
        <div className="lg:col-span-3">
          <Card variant="gemini">
            <CardHeader>
              <CardTitle className="text-white">
                {controlSections.find(s => s.id === activeSection)?.title} Controls
              </CardTitle>
            </CardHeader>
            <CardContent>
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeSection}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  {activeSection === 'display' && renderDisplayControls()}
                  {activeSection === 'layout' && renderLayoutControls()}
                  {activeSection === 'sections' && renderSectionControls()}
                  {activeSection === 'alerts' && renderAlertControls()}
                  {activeSection === 'performance' && renderPerformanceControls()}
                </motion.div>
              </AnimatePresence>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Quick Actions */}
      <Card variant="rainbow">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Quick Actions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            <Button variant="neon" onClick={() => updateSetting('darkMode', !(currentSettings?.darkMode || false))}>
              <SunMoon className="h-4 w-4 mr-2" />
              Toggle Theme
            </Button>
            <Button variant="glass" onClick={() => updateSetting('notifications', !(currentSettings?.notifications || false))}>
              {currentSettings?.notifications ? <Bell className="h-4 w-4 mr-2" /> : <BellOff className="h-4 w-4 mr-2" />}
              {currentSettings?.notifications ? 'Disable' : 'Enable'} Alerts
            </Button>
            <Button variant="ghost" onClick={() => updateSetting('soundEnabled', !(currentSettings?.soundEnabled || false))}>
              {currentSettings?.soundEnabled ? <Volume2 className="h-4 w-4 mr-2" /> : <VolumeX className="h-4 w-4 mr-2" />}
              {currentSettings?.soundEnabled ? 'Mute' : 'Unmute'} Sounds
            </Button>
            <Button variant="rainbow" onClick={() => updateSetting('fullscreenMode', !(currentSettings?.fullscreenMode || false))}>
              <Maximize2 className="h-4 w-4 mr-2" />
              Toggle Fullscreen
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};