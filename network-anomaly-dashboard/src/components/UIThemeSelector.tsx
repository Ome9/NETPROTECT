'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Palette, Sparkles, Zap, Moon, Sun, Eye } from 'lucide-react';
import { SparklesCore } from './aceternity/sparkles';

interface Theme {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  colors: {
    primary: string;
    secondary: string;
    accent: string;
    background: string;
  };
  library: string;
}

const themes: Theme[] = [
  {
    id: 'cyberpunk',
    name: 'Cyberpunk',
    description: 'Neon-infused cyberpunk aesthetics',
    icon: <Zap className="w-4 h-4" />,
    colors: {
      primary: '#ff00ff',
      secondary: '#00ffff',
      accent: '#ff0080',
      background: 'linear-gradient(135deg, #1a0033, #330066, #660033)'
    },
    library: 'DaisyUI + Aceternity'
  },
  {
    id: 'glass',
    name: 'Glass',
    description: 'Modern glassmorphism design',
    icon: <Eye className="w-4 h-4" />,
    colors: {
      primary: '#8b5cf6',
      secondary: '#06b6d4',
      accent: '#ec4899',
      background: 'linear-gradient(135deg, #667eea, #764ba2)'
    },
    library: 'UIverse + ShadcnUI'
  },
  {
    id: 'neon',
    name: 'Neon',
    description: 'Bright neon glow effects',
    icon: <Sparkles className="w-4 h-4" />,
    colors: {
      primary: '#00ff88',
      secondary: '#0088ff',
      accent: '#ff8800',
      background: 'linear-gradient(135deg, #0f172a, #1e293b, #334155)'
    },
    library: 'Aceternity + FloatUI'
  },
  {
    id: 'default',
    name: 'Default Dark',
    description: 'Clean professional dark theme',
    icon: <Moon className="w-4 h-4" />,
    colors: {
      primary: '#6366f1',
      secondary: '#8b5cf6',
      accent: '#06b6d4',
      background: 'linear-gradient(135deg, #1f2937, #374151, #4b5563)'
    },
    library: 'ShadcnUI'
  },
  {
    id: 'light',
    name: 'Light Mode',
    description: 'Clean professional light theme',
    icon: <Sun className="w-4 h-4" />,
    colors: {
      primary: '#6366f1',
      secondary: '#8b5cf6',
      accent: '#06b6d4',
      background: 'linear-gradient(135deg, #f8fafc, #e2e8f0, #cbd5e1)'
    },
    library: 'ShadcnUI'
  }
];

export const UIThemeSelector: React.FC = () => {
  const [currentTheme, setCurrentTheme] = useState('default');
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    // Apply theme to document
    const root = document.documentElement;
    const theme = themes.find(t => t.id === currentTheme) || themes[3];
    
    root.style.setProperty('--theme-primary', theme.colors.primary);
    root.style.setProperty('--theme-secondary', theme.colors.secondary);
    root.style.setProperty('--theme-accent', theme.colors.accent);
    root.style.setProperty('--theme-background', theme.colors.background);
    
    // Set DaisyUI theme attribute
    document.documentElement.setAttribute('data-theme', currentTheme);
    
    // Handle light/dark mode classes
    if (currentTheme === 'light') {
      root.classList.remove('dark');
      root.classList.add('light');
      root.style.setProperty('--theme-text', '#1f2937');
      root.style.setProperty('--theme-text-secondary', '#4b5563');
      root.style.setProperty('--theme-card-bg', 'rgba(255, 255, 255, 0.9)');
      root.style.setProperty('--theme-border', '#e5e7eb');
    } else {
      root.classList.remove('light');
      root.classList.add('dark');
      root.style.setProperty('--theme-text', '#ffffff');
      root.style.setProperty('--theme-text-secondary', '#d1d5db');
      root.style.setProperty('--theme-card-bg', 'rgba(0, 0, 0, 0.3)');
      root.style.setProperty('--theme-border', theme.colors.primary);
    }
  }, [currentTheme]);

  return (
    <div className="relative">
      <motion.button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 px-3 py-2 rounded-lg bg-gray-800/50 border border-gray-700 hover:border-purple-500 transition-all duration-300 text-gray-300 hover:text-white"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <Palette className="w-4 h-4 text-purple-400" />
        <span className="text-sm font-medium hidden sm:inline">Themes</span>
      </motion.button>

      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
              className="fixed inset-0 z-[60] bg-black/20 backdrop-blur-sm"
            />

            {/* Theme Selector Panel */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: -10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: -10 }}
              className={`absolute top-full mt-2 right-0 z-[61] w-80 max-w-[90vw] sm:max-w-80 backdrop-blur-xl rounded-xl shadow-2xl overflow-hidden ${
                currentTheme === 'light' 
                  ? 'bg-white/95 border border-gray-200' 
                  : 'bg-gray-900/95 border border-gray-800'
              }`}
            >
              {/* Header */}
              <div className={`p-4 border-b bg-gradient-to-r from-purple-600/20 to-pink-600/20 ${
                currentTheme === 'light' ? 'border-gray-200' : 'border-gray-800'
              }`}>
                <div className="relative">
                  <SparklesCore
                    id="theme-selector-sparkles"
                    background="transparent"
                    particleSize={0.8}
                    particleDensity={50}
                    className="absolute inset-0"
                    particleColor="#a855f7"
                  />
                  <h3 className={`relative text-lg font-semibold ${
                    currentTheme === 'light' ? 'text-gray-900' : 'text-white'
                  }`}>UI Library Themes</h3>
                  <p className={`relative text-sm mt-1 ${
                    currentTheme === 'light' ? 'text-gray-600' : 'text-gray-400'
                  }`}>Choose your dashboard experience</p>
                </div>
              </div>

              {/* Theme Grid */}
              <div className="p-3 space-y-2 max-h-96 overflow-y-auto">
                {themes.map((theme) => (
                  <motion.div
                    key={theme.id}
                    onClick={() => {
                      setCurrentTheme(theme.id);
                      setIsOpen(false);
                    }}
                    className={`group cursor-pointer p-3 rounded-lg border transition-all duration-300 ${
                      currentTheme === theme.id
                        ? 'border-purple-500 bg-purple-500/10 shadow-lg shadow-purple-500/20'
                        : currentTheme === 'light'
                        ? 'border-gray-300 bg-gray-50 hover:border-purple-400 hover:bg-gray-100'
                        : 'border-gray-700 bg-gray-800/30 hover:border-purple-400 hover:bg-gray-700/50'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-3">
                        <div 
                          className="w-8 h-8 rounded-full flex items-center justify-center"
                          style={{ 
                            background: `linear-gradient(45deg, ${theme.colors.primary}, ${theme.colors.accent})`
                          }}
                        >
                          {theme.icon}
                        </div>
                        <div>
                          <h4 className={`font-medium group-hover:text-purple-300 transition-colors ${
                            currentTheme === 'light' ? 'text-gray-900' : 'text-white'
                          }`}>
                            {theme.name}
                          </h4>
                          <p className={`text-xs mt-1 ${
                            currentTheme === 'light' ? 'text-gray-600' : 'text-gray-400'
                          }`}>{theme.description}</p>
                          <div className="text-xs text-purple-400 mt-1 font-medium">
                            {theme.library}
                          </div>
                        </div>
                      </div>
                      
                      {/* Color Preview */}
                      <div className="flex space-x-1">
                        <div 
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: theme.colors.primary }}
                        />
                        <div 
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: theme.colors.secondary }}
                        />
                        <div 
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: theme.colors.accent }}
                        />
                      </div>
                    </div>

                    {/* Active indicator */}
                    {currentTheme === theme.id && (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="absolute -top-1 -right-1 w-4 h-4 bg-purple-500 rounded-full flex items-center justify-center"
                      >
                        <Sun className="w-2 h-2 text-white" />
                      </motion.div>
                    )}
                  </motion.div>
                ))}
              </div>

              {/* Footer */}
              <div className={`p-3 border-t ${
                currentTheme === 'light' 
                  ? 'border-gray-200 bg-gray-50/50' 
                  : 'border-gray-800 bg-gray-900/50'
              }`}>
                <p className={`text-xs text-center ${
                  currentTheme === 'light' ? 'text-gray-500' : 'text-gray-500'
                }`}>
                  Themes combine components from DaisyUI, ShadcnUI, Aceternity UI, FloatUI & UIverse
                </p>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
};