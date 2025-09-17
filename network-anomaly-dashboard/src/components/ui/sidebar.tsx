'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Menu, X, Home, Shield, Brain, BarChart3, Settings, 
  Activity, Network, AlertTriangle, Database,
  Bell, User, HelpCircle, LogOut
} from 'lucide-react';

interface SidebarProps {
  currentView: string;
  onViewChange: (view: string) => void;
  isConnected: boolean;
  threatCount: number;
}

export const Sidebar: React.FC<SidebarProps> = ({ 
  currentView, 
  onViewChange, 
  isConnected, 
  threatCount 
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const navigationItems = [
    { id: 'overview', label: 'Overview', icon: Home, badge: null },
    { id: 'topology', label: 'Network Topology', icon: Network, badge: null },
    { id: 'threats', label: 'Threat Detection', icon: AlertTriangle, badge: threatCount > 0 ? threatCount : null },
    { id: 'model', label: 'ML Monitoring', icon: Brain, badge: null },
    { id: 'traffic', label: 'Traffic Analysis', icon: BarChart3, badge: null },
    { id: 'config', label: 'Configuration', icon: Settings, badge: null },
  ];

  const systemItems = [
    { id: 'alerts', label: 'System Alerts', icon: Bell, badge: 3 },
    { id: 'database', label: 'Database', icon: Database, badge: null },
    { id: 'performance', label: 'Performance', icon: Activity, badge: null },
  ];

  const userItems = [
    { id: 'profile', label: 'Profile', icon: User, badge: null },
    { id: 'help', label: 'Help & Support', icon: HelpCircle, badge: null },
    { id: 'logout', label: 'Logout', icon: LogOut, badge: null },
  ];

  const handleItemClick = (itemId: string) => {
    if (['overview', 'demo', 'topology', 'threats', 'model', 'traffic', 'config'].includes(itemId)) {
      onViewChange(itemId);
      setIsOpen(false);
    } else {
      console.log('Sidebar action:', itemId);
    }
  };

  return (
    <>
      {/* Enhanced Sidebar Toggle Button */}
      <motion.button
        className="fixed top-6 left-6 z-60 w-12 h-12 rounded-xl bg-black/80 backdrop-blur-xl border border-white/10 text-white cursor-pointer transition-all duration-300 flex items-center justify-center group hover:bg-pink-500/20 hover:border-pink-500/50"
        onClick={() => setIsOpen(!isOpen)}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        style={{
          left: isOpen ? '340px' : '24px',
          transition: 'left 0.4s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.3s ease, border-color 0.3s ease'
        }}
      >
        <AnimatePresence mode="wait">
          {isOpen ? (
            <motion.div
              key="close"
              initial={{ rotate: -90, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: 90, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <X className="h-5 w-5" />
            </motion.div>
          ) : (
            <motion.div
              key="menu"
              initial={{ rotate: 90, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: -90, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <Menu className="h-5 w-5" />
            </motion.div>
          )}
        </AnimatePresence>
        
        {/* Hover glow effect */}
        <div className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
             style={{
               background: 'linear-gradient(135deg, rgba(255, 0, 150, 0.1), rgba(0, 255, 255, 0.1))',
               boxShadow: '0 0 20px rgba(255, 0, 150, 0.3)'
             }} />
      </motion.button>

      {/* Enhanced Background Blur Overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="fixed inset-0 z-40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsOpen(false)}
            style={{
              background: 'radial-gradient(circle at center, rgba(0, 0, 0, 0.4) 0%, rgba(0, 0, 0, 0.6) 100%)',
              backdropFilter: 'blur(12px) saturate(1.2)'
            }}
          />
        )}
      </AnimatePresence>

      {/* Enhanced Sidebar */}
      <motion.div
        className="fixed top-0 left-0 w-80 h-full z-50 overflow-y-auto"
        initial={{ x: -320 }}
        animate={{ x: isOpen ? 0 : -320 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
        style={{
          background: 'rgba(10, 10, 10, 0.95)',
          backdropFilter: 'blur(20px)',
          borderRight: '1px solid rgba(255, 255, 255, 0.1)'
        }}
      >
        {/* Gradient accent border */}
        <div 
          className="absolute top-0 right-0 w-0.5 h-full"
          style={{
            background: 'linear-gradient(180deg, rgba(255, 0, 150, 0.8), rgba(0, 255, 255, 0.8), rgba(255, 255, 0, 0.8), rgba(255, 0, 150, 0.8))',
            backgroundSize: '100% 400%',
            animation: 'gradient-shift 4s ease infinite'
          }}
        />

        {/* Enhanced Sidebar Header */}
        <div className="p-6 border-b border-white/10"
             style={{
               background: 'linear-gradient(135deg, rgba(255, 0, 150, 0.1), rgba(0, 255, 255, 0.1))'
             }}>
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center relative overflow-hidden"
                 style={{
                   background: 'linear-gradient(135deg, #ff0096, #00ffff)',
                   boxShadow: '0 0 20px rgba(255, 0, 150, 0.4)'
                 }}>
              <Shield className="w-7 h-7 text-white relative z-10" />
              <div className="absolute inset-0 bg-gradient-to-br from-pink-500/20 to-cyan-500/20 animate-pulse" />
            </div>
            <div>
              <motion.h2 
                className="text-xl font-bold text-white"
                animate={{ 
                  backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"] 
                }}
                transition={{ 
                  duration: 3, 
                  repeat: Infinity, 
                  ease: "linear" 
                }}
              >
                NetProtect AI
              </motion.h2>
              <div className="flex items-center gap-2 mt-1">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
                <span className="text-sm text-gray-400">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Navigation Items */}
        <nav className="p-4">
          <div className="mb-8">
            <h3 className="px-3 mb-4 text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Main Navigation
            </h3>
            <div className="space-y-1">
              {navigationItems.map((item, index) => {
                const Icon = item.icon;
                const isActive = currentView === item.id;
                
                return (
                  <motion.a
                    key={item.id}
                    className={`flex items-center gap-3 px-3 py-3 rounded-xl text-gray-300 transition-all duration-300 relative overflow-hidden group cursor-pointer ${
                      isActive ? 'text-white bg-gradient-to-r from-pink-500/20 to-cyan-500/20 border-l-2 border-pink-500' : ''
                    }`}
                    onClick={() => handleItemClick(item.id)}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    whileHover={{ x: 5 }}
                  >
                    {/* Hover gradient background */}
                    <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
                         style={{
                           background: 'linear-gradient(90deg, rgba(255, 0, 150, 0.1), rgba(0, 255, 255, 0.1))'
                         }} />
                    
                    <Icon className="w-5 h-5 relative z-10" />
                    <span className="flex-1 relative z-10">{item.label}</span>
                    {item.badge && (
                      <motion.span 
                        className="px-2 py-1 text-xs font-bold text-white rounded-full relative z-10"
                        style={{
                          background: 'linear-gradient(135deg, #ff0096, #ff4757)',
                          boxShadow: '0 0 10px rgba(255, 0, 150, 0.5)'
                        }}
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{ duration: 2, repeat: Infinity }}
                      >
                        {item.badge}
                      </motion.span>
                    )}
                  </motion.a>
                );
              })}
            </div>
          </div>

          <div className="mb-8">
            <h3 className="px-3 mb-4 text-xs font-semibold text-gray-500 uppercase tracking-wider">
              System
            </h3>
            <div className="space-y-1">
              {systemItems.map((item, index) => {
                const Icon = item.icon;
                
                return (
                  <motion.a
                    key={item.id}
                    className="flex items-center gap-3 px-3 py-3 rounded-xl text-gray-300 transition-all duration-300 relative overflow-hidden group cursor-pointer"
                    onClick={() => handleItemClick(item.id)}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: (navigationItems.length + index) * 0.05 }}
                    whileHover={{ x: 5 }}
                  >
                    {/* Hover gradient background */}
                    <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
                         style={{
                           background: 'linear-gradient(90deg, rgba(255, 0, 150, 0.1), rgba(0, 255, 255, 0.1))'
                         }} />
                    
                    <Icon className="w-5 h-5 relative z-10" />
                    <span className="flex-1 relative z-10">{item.label}</span>
                    {item.badge && (
                      <motion.span 
                        className="px-2 py-1 text-xs font-bold text-white rounded-full relative z-10"
                        style={{
                          background: 'linear-gradient(135deg, #ff0096, #ff4757)',
                          boxShadow: '0 0 10px rgba(255, 0, 150, 0.5)'
                        }}
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{ duration: 2, repeat: Infinity }}
                      >
                        {item.badge}
                      </motion.span>
                    )}
                  </motion.a>
                );
              })}
            </div>
          </div>

          <div className="border-t border-white/10 pt-6">
            <h3 className="px-3 mb-4 text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Account
            </h3>
            <div className="space-y-1">
              {userItems.map((item, index) => {
                const Icon = item.icon;
                
                return (
                  <motion.a
                    key={item.id}
                    className="flex items-center gap-3 px-3 py-3 rounded-xl text-gray-300 transition-all duration-300 relative overflow-hidden group cursor-pointer"
                    onClick={() => handleItemClick(item.id)}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: (navigationItems.length + systemItems.length + index) * 0.05 }}
                    whileHover={{ x: 5 }}
                  >
                    {/* Hover gradient background */}
                    <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
                         style={{
                           background: 'linear-gradient(90deg, rgba(255, 0, 150, 0.1), rgba(0, 255, 255, 0.1))'
                         }} />
                    
                    <Icon className="w-5 h-5 relative z-10" />
                    <span className="flex-1 relative z-10">{item.label}</span>
                  </motion.a>
                );
              })}
            </div>
          </div>
        </nav>

        {/* Enhanced Sidebar Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-6 border-t border-white/10"
             style={{
               background: 'linear-gradient(135deg, rgba(255, 0, 150, 0.05), rgba(0, 255, 255, 0.05))'
             }}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 text-sm text-gray-400">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
              <span>System {isConnected ? 'Online' : 'Offline'}</span>
            </div>
            <div className="text-xs text-gray-500">
              v2.1.3
            </div>
          </div>
        </div>
      </motion.div>
    </>
  );
};