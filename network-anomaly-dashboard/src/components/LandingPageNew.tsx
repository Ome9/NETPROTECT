'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Shield, 
  Brain, 
  Network, 
  Zap, 
  Eye, 
  Lock,
  TrendingUp,
  Activity,
  Globe,
  AlertTriangle,
  CheckCircle2,
  ArrowRight,
  Cpu,
  Database,
  Wifi,
  Users,
  Target,
  BarChart3,
  Layers,
  Server
} from 'lucide-react';

interface LandingPageProps {
  systemMetrics: {
    cpuUsage: number;
    memoryUsage: number;
    networkLoad: number;
    diskUsage: number;
    gpuUsage: number;
    threatsBlocked: number;
    activeConnections: number;
    modelAccuracy: number;
  };
  networkData?: any;
  mlMetrics?: any;
  onViewChange: (view: string) => void;
}

export const LandingPageNew: React.FC<LandingPageProps> = ({ 
  systemMetrics, 
  networkData, 
  mlMetrics,
  onViewChange 
}) => {
  const [currentFeature, setCurrentFeature] = useState(0);
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);

  const features = [
    {
      icon: Brain,
      title: "AI-Powered Detection",
      description: "Advanced machine learning algorithms trained on NSL-KDD, CSE-CIC, and UNSW datasets",
      color: "purple",
      stats: "95.17% Accuracy"
    },
    {
      icon: Eye,
      title: "Real-Time Monitoring",
      description: "Live network traffic analysis with millisecond response times",
      color: "blue",
      stats: "24/7 Surveillance"
    },
    {
      icon: Shield,
      title: "Multi-Layer Protection",
      description: "Comprehensive security across network, application, and system layers",
      color: "green",
      stats: `${systemMetrics.threatsBlocked}+ Threats Blocked`
    },
    {
      icon: Zap,
      title: "Lightning Performance",
      description: "High-speed processing with minimal system impact",
      color: "yellow",
      stats: "< 10ms Latency"
    }
  ];

  const capabilities = [
    { name: "Anomaly Detection", status: "active", accuracy: 95.17 },
    { name: "Traffic Analysis", status: "active", accuracy: 92.43 },
    { name: "Threat Classification", status: "active", accuracy: 89.21 },
    { name: "Behavioral Analysis", status: "active", accuracy: 91.84 },
    { name: "Pattern Recognition", status: "active", accuracy: 93.67 },
    { name: "Risk Assessment", status: "active", accuracy: 88.92 }
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentFeature((prev) => (prev + 1) % features.length);
    }, 4000);
    return () => clearInterval(interval);
  }, [features.length]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(120,119,198,0.1),rgba(255,255,255,0.0))]" />
        <div className="absolute inset-0 bg-[conic-gradient(from_0deg,transparent,rgba(34,211,238,0.1),transparent)]" />
      </div>

      {/* Static Floating Particles - Fixed positions to prevent hydration errors */}
      <div className="absolute inset-0">
        {[
          { left: 10, top: 20 }, { left: 80, top: 15 }, { left: 25, top: 70 }, { left: 65, top: 45 },
          { left: 90, top: 80 }, { left: 15, top: 90 }, { left: 50, top: 10 }, { left: 75, top: 60 },
          { left: 5, top: 50 }, { left: 95, top: 30 }, { left: 35, top: 85 }, { left: 60, top: 25 },
          { left: 20, top: 40 }, { left: 85, top: 70 }, { left: 45, top: 95 }, { left: 70, top: 5 },
          { left: 30, top: 65 }, { left: 55, top: 35 }, { left: 40, top: 80 }, { left: 75, top: 50 }
        ].map((particle, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400 rounded-full opacity-30"
            animate={{
              x: [0, 100, 0],
              y: [0, -100, 0],
              opacity: [0.3, 0.8, 0.3]
            }}
            transition={{
              duration: 8 + i * 0.5,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            style={{
              left: `${particle.left}%`,
              top: `${particle.top}%`
            }}
          />
        ))}
      </div>

      <div className="relative z-10 container mx-auto px-6 py-12">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
          className="text-center mb-16"
        >
          <motion.div
            className="inline-flex items-center gap-3 mb-6"
            whileHover={{ scale: 1.05 }}
          >
            <Shield className="h-12 w-12 text-cyan-400" />
            <span className="text-5xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              NetProtect AI
            </span>
          </motion.div>
          
          <h1 className="text-6xl md:text-7xl font-bold text-white mb-6 leading-tight">
            Advanced Network
            <br />
            <span className="bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent">
              Security Intelligence
            </span>
          </h1>
          
          <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Harness the power of artificial intelligence to protect your network infrastructure with 
            real-time threat detection, advanced anomaly analysis, and predictive security intelligence.
          </p>

          {/* Live Status Indicators */}
          <div className="flex justify-center gap-6 mt-8">
            <motion.div 
              className="flex items-center gap-2"
              animate={{ scale: [1, 1.05, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse" />
              <span className="text-green-400 font-medium">System Online</span>
            </motion.div>
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-blue-400" />
              <span className="text-blue-400 font-medium">{systemMetrics.activeConnections} Active Connections</span>
            </div>
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-purple-400" />
              <span className="text-purple-400 font-medium">{systemMetrics.threatsBlocked}+ Threats Blocked</span>
            </div>
          </div>
        </motion.div>

        {/* Feature Spotlight */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.3 }}
          className="mb-16"
        >
          <Card variant="glass" className="bg-black/20 border-cyan-500/30 p-8">
            <AnimatePresence mode="wait">
              <motion.div
                key={currentFeature}
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -50 }}
                transition={{ duration: 0.5 }}
                className="flex items-center gap-8"
              >
                <div className={`p-6 rounded-2xl bg-gradient-to-br from-${features[currentFeature].color}-500/20 to-${features[currentFeature].color}-600/20 border border-${features[currentFeature].color}-500/30`}>
                  {React.createElement(features[currentFeature].icon, { className: `h-16 w-16 text-${features[currentFeature].color}-400` })}
                </div>
                <div className="flex-1">
                  <h3 className="text-3xl font-bold text-white mb-3">{features[currentFeature].title}</h3>
                  <p className="text-lg text-gray-300 mb-4">{features[currentFeature].description}</p>
                  <Badge variant="neon" className="text-lg px-4 py-2">
                    {features[currentFeature].stats}
                  </Badge>
                </div>
              </motion.div>
            </AnimatePresence>
          </Card>
        </motion.div>

        {/* Core Capabilities */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.5 }}
          className="mb-16"
        >
          <h2 className="text-4xl font-bold text-center text-white mb-12">
            Core Security Capabilities
          </h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {capabilities.map((capability, index) => (
              <motion.div
                key={capability.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 * index }}
                whileHover={{ scale: 1.02 }}
                onHoverStart={() => setHoveredCard(capability.name)}
                onHoverEnd={() => setHoveredCard(null)}
              >
                <Card 
                  variant="glass" 
                  className={`h-full transition-all duration-300 ${
                    hoveredCard === capability.name 
                      ? 'border-cyan-400/50 bg-cyan-500/10 shadow-lg shadow-cyan-500/25' 
                      : 'border-gray-700/50 bg-black/20'
                  }`}
                >
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-semibold text-white">{capability.name}</h3>
                      <CheckCircle2 className="h-5 w-5 text-green-400" />
                    </div>
                    <div className="space-y-3">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-300">Accuracy</span>
                        <span className="text-green-400 font-medium">{capability.accuracy}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <motion.div
                          className="bg-gradient-to-r from-green-500 to-cyan-500 h-2 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${capability.accuracy}%` }}
                          transition={{ duration: 1, delay: 0.2 * index }}
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Technology Stack */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.7 }}
          className="mb-16"
        >
          <h2 className="text-4xl font-bold text-center text-white mb-12">
            Advanced Technology Stack
          </h2>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { icon: Database, title: "Multi-Dataset Training", desc: "NSL-KDD, CSE-CIC, UNSW datasets", color: "purple" },
              { icon: Brain, title: "Deep Learning", desc: "Neural networks & ensemble methods", color: "blue" },
              { icon: Zap, title: "Real-Time Processing", desc: "Stream processing & edge computing", color: "yellow" },
              { icon: Network, title: "Network Intelligence", desc: "Traffic analysis & behavioral modeling", color: "green" }
            ].map((tech, index) => (
              <motion.div
                key={tech.title}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 0.1 * index }}
                whileHover={{ y: -5 }}
              >
                <Card variant="glass" className="text-center p-6 h-full bg-black/20 border-gray-700/50 hover:border-cyan-400/50 transition-all duration-300">
                  <div className={`mx-auto w-16 h-16 rounded-full bg-gradient-to-br from-${tech.color}-500/20 to-${tech.color}-600/20 flex items-center justify-center mb-4`}>
                    {React.createElement(tech.icon, { className: `h-8 w-8 text-${tech.color}-400` })}
                  </div>
                  <h3 className="font-bold text-white mb-2">{tech.title}</h3>
                  <p className="text-sm text-gray-300">{tech.desc}</p>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.9 }}
          className="text-center"
        >
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Button
              size="lg"
              variant="neon"
              onClick={() => onViewChange('overview')}
              className="group px-8 py-4 text-lg font-semibold"
            >
              <Activity className="h-5 w-5 mr-3 group-hover:animate-pulse" />
              Launch Dashboard
              <ArrowRight className="h-5 w-5 ml-3 group-hover:translate-x-1 transition-transform" />
            </Button>
            
            <Button
              size="lg"
              variant="outline"
              onClick={() => onViewChange('topology')}
              className="px-8 py-4 text-lg font-semibold border-cyan-400 text-cyan-400 hover:bg-cyan-400/10"
            >
              <Network className="h-5 w-5 mr-3" />
              View Network
            </Button>
          </div>

          <p className="text-gray-400 mt-6 text-sm">
            Experience next-generation network security powered by artificial intelligence
          </p>
        </motion.div>
      </div>
    </div>
  );
};