'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Plus, Zap, Settings, Bell } from 'lucide-react';

interface FabProps {
  icon?: 'plus' | 'zap' | 'settings' | 'bell' | React.ReactNode;
  onClick?: () => void;
  className?: string;
  variant?: 'default' | 'rainbow' | 'neon';
  size?: 'sm' | 'md' | 'lg';
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
  tooltip?: string;
}

export const Fab: React.FC<FabProps> = ({
  icon = 'plus',
  onClick,
  className,
  variant = 'default',
  size = 'md',
  position = 'bottom-right',
  tooltip
}) => {
  const getIcon = () => {
    if (React.isValidElement(icon)) {
      return icon;
    }
    
    switch (icon) {
      case 'plus':
        return <Plus className="h-6 w-6" />;
      case 'zap':
        return <Zap className="h-6 w-6" />;
      case 'settings':
        return <Settings className="h-6 w-6" />;
      case 'bell':
        return <Bell className="h-6 w-6" />;
      default:
        return <Plus className="h-6 w-6" />;
    }
  };

  const getSizeClasses = () => {
    switch (size) {
      case 'sm':
        return 'w-12 h-12';
      case 'lg':
        return 'w-16 h-16';
      default:
        return 'w-14 h-14';
    }
  };

  const getPositionClasses = () => {
    switch (position) {
      case 'bottom-left':
        return 'bottom-6 left-6';
      case 'top-right':
        return 'top-6 right-6';
      case 'top-left':
        return 'top-6 left-6';
      default:
        return 'bottom-6 right-6';
    }
  };

  const getVariantStyle = () => {
    switch (variant) {
      case 'rainbow':
        return {
          background: 'linear-gradient(135deg, #ff0096, #00ffff)',
          boxShadow: '0 8px 24px rgba(0, 0, 0, 0.3), 0 0 20px rgba(255, 0, 150, 0.4)'
        };
      case 'neon':
        return {
          background: '#3b82f6',
          boxShadow: '0 8px 24px rgba(0, 0, 0, 0.3), 0 0 20px rgba(59, 130, 246, 0.5)'
        };
      default:
        return {
          background: '#374151',
          boxShadow: '0 8px 24px rgba(0, 0, 0, 0.3)'
        };
    }
  };

  return (
    <motion.button
      className={cn(
        'fixed z-50 rounded-full border-none text-white cursor-pointer flex items-center justify-center transition-all duration-300',
        getSizeClasses(),
        getPositionClasses(),
        className
      )}
      onClick={onClick}
      style={getVariantStyle()}
      whileHover={{ 
        scale: 1.1, 
        rotate: variant === 'rainbow' ? 90 : 0 
      }}
      whileTap={{ scale: 0.9 }}
      initial={{ scale: 0, rotate: -180 }}
      animate={{ scale: 1, rotate: 0 }}
      transition={{ 
        type: 'spring', 
        stiffness: 300, 
        damping: 20,
        delay: 0.5
      }}
      title={tooltip}
    >
      <motion.div
        animate={{ rotate: variant === 'rainbow' ? [0, 360] : 0 }}
        transition={{ 
          duration: 2, 
          repeat: variant === 'rainbow' ? Infinity : 0, 
          ease: 'linear' 
        }}
      >
        {getIcon()}
      </motion.div>
    </motion.button>
  );
};