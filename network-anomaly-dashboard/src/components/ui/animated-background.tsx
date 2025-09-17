"use client"

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface AnimatedBackgroundProps {
  children: React.ReactNode;
  className?: string;
}

interface Particle {
  id: number;
  initialX: number;
  initialY: number;
  targetX: number;
  targetY: number;
  duration: number;
}

export const AnimatedBackground: React.FC<AnimatedBackgroundProps> = ({ 
  children, 
  className = "" 
}) => {
  const [particles, setParticles] = useState<Particle[]>([]);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const width = window.innerWidth;
    const height = window.innerHeight;
    
    const newParticles = [...Array(20)].map((_, i) => ({
      id: i,
      initialX: Math.random() * width,
      initialY: Math.random() * height,
      targetX: Math.random() * width,
      targetY: Math.random() * height,
      duration: Math.random() * 10 + 10
    }));
    
    setParticles(newParticles);
  }, []);

  if (!mounted) {
    return (
      <div className={`relative min-h-screen overflow-hidden ${className}`}>
        <div className="absolute inset-0 cyber-grid opacity-30" />
        <div className="relative z-10">
          {children}
        </div>
      </div>
    );
  }

  return (
    <div className={`relative min-h-screen overflow-hidden ${className}`}>
      {/* Animated grid background */}
      <div className="absolute inset-0 cyber-grid opacity-30" />
      
      {/* Floating particles */}
      <div className="absolute inset-0">
        {particles.map((particle) => (
          <motion.div
            key={particle.id}
            className="absolute w-1 h-1 bg-blue-400 rounded-full"
            initial={{
              x: particle.initialX,
              y: particle.initialY,
              opacity: 0
            }}
            animate={{
              x: particle.targetX,
              y: particle.targetY,
              opacity: [0, 1, 0]
            }}
            transition={{
              duration: particle.duration,
              repeat: Infinity,
              ease: "linear"
            }}
          />
        ))}
      </div>

      {/* Rainbow gradient orbs */}
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          className="absolute -top-40 -left-40 w-80 h-80 rainbow-gradient rounded-full opacity-20 blur-3xl"
          animate={{
            x: [0, 100, 0],
            y: [0, 50, 0],
            scale: [1, 1.2, 1]
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        <motion.div
          className="absolute -top-40 -right-40 w-80 h-80 rainbow-gradient rounded-full opacity-20 blur-3xl"
          animate={{
            x: [0, -100, 0],
            y: [0, 100, 0],
            scale: [1.2, 1, 1.2]
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        <motion.div
          className="absolute -bottom-40 -left-40 w-80 h-80 rainbow-gradient rounded-full opacity-20 blur-3xl"
          animate={{
            x: [0, 150, 0],
            y: [0, -50, 0],
            scale: [1, 1.3, 1]
          }}
          transition={{
            duration: 30,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      </div>

      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
};