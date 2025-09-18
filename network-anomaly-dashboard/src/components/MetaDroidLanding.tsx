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

interface MetaDroidLandingProps {
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

// Animation variants inspired by meta-droid
const staggerContainer = {
  hidden: {},
  show: {
    transition: {
      staggerChildren: 0.2,
      delayChildren: 0.1,
    },
  },
};

const textVariant = (delay: number) => ({
  hidden: {
    y: -50,
    opacity: 0,
  },
  show: {
    y: 0,
    opacity: 1,
    transition: {
      type: 'spring' as const,
      duration: 1.25,
      delay,
    },
  },
});

const slideIn = (direction: string, type: string, delay: number, duration: number) => ({
  hidden: {
    x: direction === 'left' ? '-100%' : direction === 'right' ? '100%' : 0,
    y: direction === 'up' ? '100%' : direction === 'down' ? '100%' : 0,
  },
  show: {
    x: 0,
    y: 0,
    transition: {
      type: type as 'spring' | 'tween',
      delay,
      duration,
      ease: 'easeOut' as const,
    },
  },
});

export const MetaDroidLanding: React.FC<MetaDroidLandingProps> = ({ 
  systemMetrics, 
  networkData, 
  mlMetrics,
  onViewChange 
}) => {
  const [currentFeature, setCurrentFeature] = useState(0);
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);

  const capabilities = [
    { name: "AI Anomaly Detection", accuracy: 95.17, color: "from-purple-500 to-pink-500" },
    { name: "Real-Time Monitoring", accuracy: 92.43, color: "from-blue-500 to-cyan-500" },
    { name: "Threat Classification", accuracy: 89.21, color: "from-green-500 to-emerald-500" },
    { name: "Pattern Recognition", accuracy: 93.67, color: "from-orange-500 to-red-500" },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentFeature((prev) => (prev + 1) % capabilities.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [capabilities.length]);

  return (
    <div className="bg-[#1A232E] overflow-hidden min-h-screen">
      {/* Hero Section - Meta-droid inspired */}
      <section className="sm:py-16 py-12 sm:px-16 px-6">
        <motion.div
          variants={staggerContainer}
          initial="hidden"
          whileInView="show"
          viewport={{ once: false, amount: 0.25 }}
          className="2xl:max-w-[2200px] w-full mx-auto flex flex-col"
        >
          <div className="relative z-10 flex flex-col items-center justify-center">
            <motion.h1
              variants={textVariant(1.1)}
              className="font-bold lg:text-[114px] md:text-[100px] sm:text-[60px] text-[44px] lg:leading-[158.4px] md:leading-[114.4px] sm:leading-[74.4px] leading-[64.4px] uppercase text-white text-center"
            >
              NetProtect
            </motion.h1>
            <motion.div
              variants={textVariant(1.2)}
              className="flex flex-row items-center justify-center"
            >
              <h1 className="font-bold lg:text-[114px] md:text-[100px] sm:text-[60px] text-[44px] lg:leading-[158.4px] md:leading-[114.4px] sm:leading-[74.4px] leading-[64.4px] uppercase text-white">
                {' '}AI{' '}
              </h1>
              <div className="md:w-[212px] sm:w-[80px] w-[60px] md:h-[90px] sm:h-[48px] h-[38px] md:border-[18px] sm:border-[8px] rounded-r-[50px] border-white sm:mx-2 mx-[6px] bg-gradient-to-r from-purple-500 to-cyan-500" />
              <h1 className="font-bold lg:text-[114px] md:text-[100px] sm:text-[60px] text-[44px] lg:leading-[158.4px] md:leading-[114.4px] sm:leading-[74.4px] leading-[64.4px] uppercase text-white">
                GUARD
              </h1>
            </motion.div>
          </div>

          <motion.div
            variants={slideIn('right', 'tween', 0.2, 1)}
            className="relative w-full lg:-mt-[30px] md:-mt-[18px] -mt-[15px] 2xl:pl-[280px]"
          >
            {/* Hero gradient background */}
            <div 
              className="absolute w-full h-[300px] rounded-tl-[140px] z-[0] sm:-top-[20px] -top-[10px]"
              style={{
                background: 'linear-gradient(97.86deg, #a509ff 0%, #34acc7 53.65%, #a134c7 100%)'
              }}
            />
            
            {/* Main hero content */}
            <div className="relative w-full sm:h-[500px] h-[350px] bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 rounded-tl-[140px] z-10 flex items-center justify-center overflow-hidden">
              {/* Animated Network Visualization */}
              <div className="absolute inset-0 opacity-20">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(165,9,255,0.1),rgba(255,255,255,0.0))]" />
                <div className="absolute inset-0 bg-[conic-gradient(from_0deg,transparent,rgba(52,172,199,0.1),transparent)]" />
              </div>
              
              {/* Network nodes animation */}
              <div className="relative z-10 flex flex-col items-center justify-center space-y-6">
                <div className="grid grid-cols-3 gap-8 items-center">
                  {[Shield, Network, Brain, Database, Eye, Zap].map((Icon, index) => (
                    <motion.div
                      key={index}
                      className="relative"
                      animate={{
                        scale: [1, 1.2, 1],
                        opacity: [0.6, 1, 0.6]
                      }}
                      transition={{
                        duration: 2,
                        delay: index * 0.3,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                    >
                      <div className="w-16 h-16 bg-white/10 rounded-full flex items-center justify-center backdrop-blur-sm border border-white/20">
                        <Icon className="h-8 w-8 text-cyan-400" />
                      </div>
                      {/* Connection lines */}
                      {index < 5 && (
                        <div className="absolute top-8 -right-4 w-8 h-0.5 bg-gradient-to-r from-cyan-400 to-purple-400 opacity-60" />
                      )}
                    </motion.div>
                  ))}
                </div>
                
                <motion.div
                  className="text-center"
                  variants={textVariant(1.5)}
                >
                  <p className="text-xl text-white/80 font-semibold">
                    Advanced Network Security Intelligence
                  </p>
                  <p className="text-lg text-cyan-300 mt-2">
                    {systemMetrics.threatsBlocked}+ Threats Blocked • {systemMetrics.activeConnections} Active Connections
                  </p>
                </motion.div>
              </div>
            </div>

            {/* Rotating badge */}
            <div className="w-full flex justify-end sm:-mt-[70px] -mt-[50px] pr-[40px] relative z-10 2xl:-ml-[100px]">
              <motion.div
                className="sm:w-[155px] w-[100px] sm:h-[155px] h-[100px] bg-gradient-to-br from-purple-500 to-cyan-500 rounded-full flex items-center justify-center"
                animate={{ rotate: 360 }}
                transition={{ repeat: Infinity, duration: 10, repeatType: 'loop' }}
              >
                <div className="text-white font-bold text-center">
                  <div className="text-2xl">{systemMetrics.modelAccuracy.toFixed(1)}%</div>
                  <div className="text-xs">ACCURACY</div>
                </div>
              </motion.div>
            </div>
          </motion.div>
        </motion.div>
      </section>

      {/* Security Capabilities - Meta-droid About section inspired */}
      <section className="relative sm:py-16 py-12 sm:px-16 px-6">
        {/* Gradient background effects */}
        <div 
          className="absolute top-0 right-0 w-[200px] h-[438px] opacity-50"
          style={{
            background: 'linear-gradient(270deg, hsl(295deg 76% 51%) 0%, hsl(284deg 70% 73%) 26%, hsl(257deg 70% 86%) 39%, hsl(202deg 92% 90%) 50%, hsl(215deg 77% 81%) 61%, hsl(221deg 73% 70%) 74%, hsl(220deg 76% 51%) 100%)',
            filter: 'blur(125px)'
          }}
        />
        
        <motion.div
          variants={staggerContainer}
          initial="hidden"
          whileInView="show"
          viewport={{ once: false, amount: 0.25 }}
          className="2xl:max-w-[1280px] w-full mx-auto"
        >
          <div className="flex lg:flex-row flex-col gap-8">
            <motion.div
              variants={textVariant(0.5)}
              className="flex-[0.5] lg:max-w-[370px] flex justify-end flex-col gradient-05 sm:p-8 p-4 rounded-[32px] border-[1px] border-[#6A6A6A] relative"
            >
              <div className="feedback-gradient" />
              <div>
                <h4 className="font-bold sm:text-[32px] text-[26px] sm:leading-[40.32px] leading-[36.32px] text-white">
                  AI-Powered
                </h4>
                <p className="mt-[8px] font-normal sm:text-[18px] text-[12px] sm:leading-[22.68px] leading-[16.68px] text-white opacity-50">
                  Advanced machine learning algorithms trained on multiple datasets
                </p>
              </div>

              <p className="mt-[24px] font-normal sm:text-[24px] text-[18px] sm:leading-[45.6px] leading-[39.6px] text-white">
                "NetProtect AI represents the next evolution in network security, 
                combining cutting-edge artificial intelligence with real-time threat detection 
                to provide unprecedented protection for modern digital infrastructure."
              </p>
            </motion.div>

            <motion.div
              variants={slideIn('right', 'tween', 0.2, 1)}
              className="relative flex-1 flex justify-center items-center"
            >
              <div className="grid grid-cols-2 gap-6 w-full max-w-[600px]">
                {capabilities.map((capability, index) => (
                  <motion.div
                    key={capability.name}
                    variants={textVariant(0.8 + index * 0.2)}
                    className="relative group"
                  >
                    <Card className="bg-white/5 border-white/10 backdrop-blur-sm hover:bg-white/10 transition-all duration-300">
                      <CardContent className="p-6">
                        <div className={`w-full h-2 rounded-full bg-gradient-to-r ${capability.color} mb-4`}>
                          <motion.div
                            className="h-full bg-white rounded-full"
                            initial={{ width: 0 }}
                            whileInView={{ width: `${capability.accuracy}%` }}
                            transition={{ duration: 1.5, delay: index * 0.2 }}
                          />
                        </div>
                        <h3 className="font-semibold text-white text-lg mb-2">
                          {capability.name}
                        </h3>
                        <p className="text-white/60 text-sm">
                          {capability.accuracy}% accuracy rate
                        </p>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>
        </motion.div>
      </section>

      {/* Call to Action */}
      <section className="sm:py-16 py-12 sm:px-16 px-6">
        <motion.div
          variants={staggerContainer}
          initial="hidden"
          whileInView="show"
          viewport={{ once: false, amount: 0.25 }}
          className="2xl:max-w-[1280px] w-full mx-auto flex flex-col"
        >
          <motion.div
            variants={textVariant(0.5)}
            className="text-center mb-12"
          >
            <h2 className="font-bold lg:text-[64px] md:text-[56px] sm:text-[48px] text-[32px] text-white mb-6">
              Ready to Secure Your Network?
            </h2>
            <p className="text-white/70 text-lg max-w-[600px] mx-auto">
              Experience the power of AI-driven network security with real-time threat detection and advanced anomaly analysis.
            </p>
          </motion.div>

          <motion.div
            variants={slideIn('up', 'tween', 0.3, 1)}
            className="flex flex-col sm:flex-row gap-6 justify-center items-center"
          >
            <Button
              size="lg"
              onClick={() => onViewChange('overview')}
              className="group px-8 py-4 text-lg font-semibold bg-gradient-to-r from-purple-500 to-cyan-500 hover:from-purple-600 hover:to-cyan-600 text-white border-0"
            >
              <Activity className="h-5 w-5 mr-3 group-hover:animate-pulse" />
              Launch Dashboard
              <ArrowRight className="h-5 w-5 ml-3 group-hover:translate-x-1 transition-transform" />
            </Button>
            
            <Button
              size="lg"
              variant="outline"
              onClick={() => onViewChange('topology')}
              className="px-8 py-4 text-lg font-semibold border-white/30 text-white hover:bg-white/10"
            >
              <Network className="h-5 w-5 mr-3" />
              Explore Network
            </Button>
          </motion.div>

          {/* Live metrics footer */}
          <motion.div
            variants={slideIn('up', 'tween', 0.5, 1)}
            className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16"
          >
            {[
              { icon: Shield, label: 'Threats Blocked', value: systemMetrics.threatsBlocked, color: 'text-green-400' },
              { icon: Activity, label: 'Active Connections', value: systemMetrics.activeConnections, color: 'text-blue-400' },
              { icon: Brain, label: 'Model Accuracy', value: `${systemMetrics.modelAccuracy.toFixed(1)}%`, color: 'text-purple-400' }
            ].map((metric, index) => (
              <div key={metric.label} className="text-center p-6 bg-white/5 rounded-2xl backdrop-blur-sm border border-white/10">
                <metric.icon className={`h-8 w-8 ${metric.color} mx-auto mb-3`} />
                <div className={`text-3xl font-bold ${metric.color} mb-2`}>
                  {typeof metric.value === 'number' ? metric.value.toLocaleString() : metric.value}
                </div>
                <div className="text-white/60 text-sm">{metric.label}</div>
              </div>
            ))}
          </motion.div>
        </motion.div>
      </section>
    </div>
  );
};