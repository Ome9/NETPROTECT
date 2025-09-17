'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { 
  Shield, 
  Activity, 
  Settings, 
  Bell, 
  User, 
  ChevronDown,
  Zap,
  Lock,
  Eye
} from 'lucide-react';

interface NavbarProps {
  currentView: string;
  onViewChange: (view: string) => void;
  isConnected: boolean;
  threatCount: number;
  userName?: string;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentView,
  onViewChange,
  isConnected,
  threatCount,
  userName = "Admin"
}) => {
  const navItems = [
    { id: 'overview', label: 'Dashboard', icon: Activity },
    { id: 'topology', label: 'Network', icon: Eye },
    { id: 'threats', label: 'Threats', icon: Shield, badge: threatCount },
    { id: 'model', label: 'AI Models', icon: Zap },
    { id: 'config', label: 'Settings', icon: Settings },
  ];

  return (
    <motion.nav
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="bg-gray-900/95 backdrop-blur-sm border-b border-gray-700 sticky top-0 z-50"
    >
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo Section */}
          <motion.div 
            className="flex items-center gap-3"
            whileHover={{ scale: 1.02 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            {/* NetProtect Logo */}
            <div className="relative">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center shadow-lg">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-gray-900 flex items-center justify-center">
                <Lock className="h-2 w-2 text-white" />
              </div>
            </div>
            <div className="flex flex-col">
              <h1 className="text-xl font-bold text-white leading-tight">NetProtect</h1>
              <p className="text-xs text-gray-400 leading-tight">AI Security Platform</p>
            </div>
          </motion.div>

          {/* Navigation Items */}
          <div className="flex items-center gap-1">
            {navItems.map((item) => (
              <motion.div key={item.id} className="relative">
                <Button
                  variant={currentView === item.id ? "neon" : "ghost"}
                  size="sm"
                  onClick={() => onViewChange(item.id)}
                  className={`flex items-center gap-2 text-sm transition-all duration-200 ${
                    currentView === item.id 
                      ? 'text-blue-400 bg-blue-500/10' 
                      : 'text-gray-300 hover:text-white hover:bg-gray-800'
                  }`}
                >
                  <item.icon className="h-4 w-4" />
                  <span className="hidden md:inline">{item.label}</span>
                  {item.badge && item.badge > 0 && (
                    <Badge variant="destructive" className="text-xs px-1.5 py-0.5 min-w-5 h-5">
                      {item.badge}
                    </Badge>
                  )}
                </Button>
              </motion.div>
            ))}
          </div>

          {/* Right Section - Status & User */}
          <div className="flex items-center gap-4">
            
            {/* System Status */}
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
              }`} />
              <span className="text-sm text-gray-300 hidden sm:inline">
                {isConnected ? 'Online' : 'Offline'}
              </span>
            </div>

            <Separator orientation="vertical" className="h-6 bg-gray-600" />

            {/* Notifications */}
            <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }}>
              <Button variant="ghost" size="sm" className="relative">
                <Bell className="h-4 w-4 text-gray-300" />
                {threatCount > 0 && (
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full flex items-center justify-center">
                    <span className="text-xs text-white font-bold">{threatCount > 9 ? '9+' : threatCount}</span>
                  </div>
                )}
              </Button>
            </motion.div>

            {/* User Profile */}
            <motion.div
              whileHover={{ scale: 1.02 }}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-800/50 border border-gray-700 cursor-pointer hover:bg-gray-800 transition-colors"
            >
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                <User className="h-4 w-4 text-white" />
              </div>
              <div className="hidden md:block">
                <p className="text-sm text-white font-medium">{userName}</p>
                <p className="text-xs text-gray-400">Administrator</p>
              </div>
              <ChevronDown className="h-4 w-4 text-gray-400" />
            </motion.div>

          </div>
        </div>
      </div>
    </motion.nav>
  );
};