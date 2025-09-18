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
  Grid, Layout, Maximize2, Minimize2, Sparkles
} from 'lucide-react';

// Enhanced Switch component with better theming
const Switch: React.FC<{
  checked?: boolean;
  onCheckedChange?: (checked: boolean) => void;
  disabled?: boolean;
  className?: string;
  variant?: 'default' | 'neon' | 'glass';
}> = ({ checked = false, onCheckedChange, disabled = false, className = '', variant = 'default' }) => {
  const variantStyles = {
    default: checked 
      ? 'bg-gradient-to-r from-blue-500 to-purple-600 shadow-lg shadow-blue-500/25' 
      : 'bg-gray-700',
    neon: checked 
      ? 'bg-gradient-to-r from-cyan-500 to-purple-500 shadow-lg shadow-cyan-500/50 border border-cyan-400' 
      : 'bg-slate-800 border border-slate-600',
    glass: checked 
      ? 'bg-white/20 backdrop-blur-lg border border-white/30 shadow-lg shadow-purple-500/25' 
      : 'bg-black/20 backdrop-blur-lg border border-white/10'
  };

  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      className={`
        peer inline-flex h-7 w-12 shrink-0 cursor-pointer items-center rounded-full 
        border-2 border-transparent transition-all duration-300 ease-in-out
        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 
        focus-visible:ring-offset-2 focus-visible:ring-offset-black 
        disabled:cursor-not-allowed disabled:opacity-50 hover:scale-105
        ${variantStyles[variant]} 
        ${className}
      `}
      onClick={() => !disabled && onCheckedChange?.(!checked)}
    >
      <div
        className={`
          pointer-events-none block h-5 w-5 rounded-full bg-white shadow-xl 
          ring-0 transition-all duration-300 ease-in-out
          ${checked ? 'translate-x-5 shadow-lg' : 'translate-x-0.5'}
          ${variant === 'neon' && checked ? 'shadow-cyan-300/50' : ''}
        `}
      />
    </button>
  );
};

// Enhanced Theme Button Component
const ThemeButton: React.FC<{
  theme: {
    name: string;
    color: string;
    accent: string;
    description: string;
  };
  isActive: boolean;
  onClick: () => void;
  variant?: 'default' | 'glass' | 'neon';
}> = ({ theme, isActive, onClick, variant = 'default' }) => {
  const variantClasses = {
    default: `bg-gradient-to-br ${theme.color} hover:scale-105 hover:shadow-xl transition-all duration-300`,
    glass: `bg-gradient-to-br ${theme.color} backdrop-blur-lg bg-opacity-20 border border-white/20 hover:border-white/40 hover:scale-105 transition-all duration-300`,
    neon: `bg-gradient-to-br ${theme.color} border-2 border-transparent hover:border-white/30 hover:shadow-2xl hover:scale-105 transition-all duration-300 relative overflow-hidden`
  };

  return (
    <div className="relative group">
      <Button
        variant="ghost"
        className={`h-20 w-full text-white relative overflow-hidden ${variantClasses[variant]} ${
          isActive ? 'ring-2 ring-white/50 shadow-lg' : ''
        }`}
        onClick={onClick}
      >
        {variant === 'neon' && (
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-pulse" />
        )}
        <div className="flex flex-col items-center justify-center relative z-10">
          <div className="text-sm font-bold mb-1">{theme.name}</div>
          <div className="text-xs opacity-80">{theme.description}</div>
        </div>
        {isActive && (
          <>
            <div className="absolute top-2 right-2 w-3 h-3 bg-white rounded-full animate-pulse shadow-lg"></div>
            <Sparkles className="absolute bottom-2 left-2 h-4 w-4 text-white/80 animate-pulse" />
          </>
        )}
      </Button>
    </div>
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
    landingPageTheme: 'modern' | 'meta-droid';
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
    
    // Call the parent's setting change handler immediately
    onSettingChange?.(key, value);
    
    // Trigger specific callbacks for external components
    if (key === 'darkMode') {
      onThemeChange?.(value ? 'dark' : 'light');
    }
    if (key === 'gridDensity' || key === 'sidebarCollapsed' || key === 'fullscreenMode' || key === 'compactMode') {
      onLayoutChange?.(key);
    }
    if (key.endsWith('Metrics') || key.endsWith('Topology') || key.endsWith('Detection') || 
        key.endsWith('Analysis') || key.endsWith('Monitoring') || key.endsWith('configuration')) {
      onSectionToggle?.(key, value as boolean);
    }
    
    // Force update notification
    console.log('✅ Setting update completed:', { key, value });
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
        <h4 className="text-white font-medium flex items-center gap-2">
          <Palette className="h-5 w-5" />
          Color Themes
        </h4>
        <div className="grid grid-cols-2 gap-4">
          {[
            { 
              name: 'Cyberpunk', 
              color: 'from-purple-900 via-pink-800 to-purple-900',
              accent: '#ff00ff',
              description: 'Electric vibes'
            },
            { 
              name: 'Ocean Blue', 
              color: 'from-blue-900 via-cyan-800 to-blue-900',
              accent: '#0ea5e9',
              description: 'Deep ocean'
            },
            { 
              name: 'Forest Green', 
              color: 'from-green-900 via-emerald-800 to-green-900',
              accent: '#10b981',
              description: 'Natural harmony'
            },
            { 
              name: 'Sunset Orange', 
              color: 'from-orange-900 via-red-800 to-orange-900',
              accent: '#f97316',
              description: 'Warm glow'
            },
            { 
              name: 'Purple Galaxy', 
              color: 'from-violet-900 via-purple-800 to-indigo-900',
              accent: '#8b5cf6',
              description: 'Cosmic energy'
            },
            { 
              name: 'Arctic Ice', 
              color: 'from-slate-800 via-blue-900 to-slate-800',
              accent: '#64748b',
              description: 'Cool elegance'
            },
            { 
              name: 'Golden Hour', 
              color: 'from-yellow-800 via-orange-700 to-amber-800',
              accent: '#f59e0b',
              description: 'Warm radiance'
            },
            { 
              name: 'Midnight', 
              color: 'from-gray-800 via-slate-700 to-zinc-600',
              accent: '#374151',
              description: 'Dark elegance'
            }
          ].map((theme, index) => (
            <ThemeButton
              key={theme.name}
              theme={theme}
              isActive={currentSettings?.colorTheme === theme.name.toLowerCase()}
              onClick={() => updateSetting('colorTheme', theme.name.toLowerCase())}
              variant={index % 3 === 0 ? 'neon' : index % 2 === 0 ? 'glass' : 'default'}
            />
          ))}
        </div>
      </div>

      {/* Landing Page Themes */}
      <div className="space-y-4">
        <h4 className="text-white font-medium flex items-center gap-2">
          <Layout className="h-5 w-5" />
          Landing Page Theme
        </h4>
        <div className="grid grid-cols-2 gap-3">
          <Button
            variant={(currentSettings?.landingPageTheme || 'modern') === 'modern' ? "neon" : "outline"}
            className="flex flex-col items-center p-4 h-auto"
            onClick={() => updateSetting('landingPageTheme', 'modern')}
          >
            <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-purple-500 rounded mb-2" />
            <span className="text-sm">Modern</span>
            <span className="text-xs text-gray-400">Cyberpunk style</span>
          </Button>
          <Button
            variant={(currentSettings?.landingPageTheme || 'modern') === 'meta-droid' ? "neon" : "outline"}
            className="flex flex-col items-center p-4 h-auto"
            onClick={() => updateSetting('landingPageTheme', 'meta-droid')}
          >
            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-blue-500 rounded mb-2" />
            <span className="text-sm">Meta-Droid</span>
            <span className="text-xs text-gray-400">Futuristic design</span>
          </Button>
        </div>
      </div>
    </div>
  );

  const renderLayoutControls = () => (
    <div className="space-y-6">
      <div className="space-y-4">
        <h4 className="text-white font-medium flex items-center gap-2">
          <Layout className="h-5 w-5" />
          Layout Options
        </h4>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Compact Mode</span>
          <Switch 
            checked={currentSettings?.compactMode || false}
            onCheckedChange={(checked: boolean) => updateSetting('compactMode', checked)}
            variant="neon"
          />
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Sidebar Collapsed</span>
          <Switch 
            checked={currentSettings?.sidebarCollapsed || false}
            onCheckedChange={(checked: boolean) => updateSetting('sidebarCollapsed', checked)}
            variant="glass"
          />
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Fullscreen Mode</span>
          <Switch 
            checked={currentSettings?.fullscreenMode || false}
            onCheckedChange={(checked: boolean) => updateSetting('fullscreenMode', checked)}
            variant="default"
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
              variant={currentSettings?.gridDensity === option.value ? "default" : "ghost"}
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
        variant="default" 
        className="w-full"
        onClick={() => {
          // Reset to default values
          updateSetting('refreshRate', 5);
          updateSetting('dataRetention', 24);
          updateSetting('realtimeUpdates', true);
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
                  variant={activeSection === section.id ? "default" : "ghost"}
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
      <Card variant="glass">
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
            <Button variant="default" onClick={() => updateSetting('fullscreenMode', !(currentSettings?.fullscreenMode || false))}>
              <Maximize2 className="h-4 w-4 mr-2" />
              Toggle Fullscreen
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};