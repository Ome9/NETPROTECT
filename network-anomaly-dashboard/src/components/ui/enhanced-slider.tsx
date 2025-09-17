'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface EnhancedSliderProps {
  min?: number;
  max?: number;
  step?: number;
  value?: number;
  defaultValue?: number;
  onChange?: (value: number) => void;
  className?: string;
  variant?: 'default' | 'rainbow' | 'neon' | 'gemini';
  glowColor?: 'blue' | 'red' | 'green' | 'purple' | 'pink' | 'cyan';
  label?: string;
  showValue?: boolean;
  disabled?: boolean;
}

export const EnhancedSlider: React.FC<EnhancedSliderProps> = ({
  min = 0,
  max = 100,
  step = 1,
  value,
  defaultValue = 0,
  onChange,
  className,
  variant = 'default',
  glowColor = 'blue',
  label,
  showValue = true,
  disabled = false
}) => {
  const [internalValue, setInternalValue] = useState(value ?? defaultValue);
  const [isDragging, setIsDragging] = useState(false);
  const [isHovering, setIsHovering] = useState(false);
  const sliderRef = useRef<HTMLDivElement>(null);
  const thumbRef = useRef<HTMLDivElement>(null);

  const currentValue = value ?? internalValue;
  const percentage = ((currentValue - min) / (max - min)) * 100;

  useEffect(() => {
    if (value !== undefined) {
      setInternalValue(value);
    }
  }, [value]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (disabled) return;
    setIsDragging(true);
    updateValue(e);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (isDragging && !disabled) {
      updateValue(e);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const updateValue = (e: MouseEvent | React.MouseEvent) => {
    if (!sliderRef.current || disabled) return;

    const rect = sliderRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    const newValue = min + (percentage / 100) * (max - min);
    const steppedValue = Math.round(newValue / step) * step;
    const clampedValue = Math.max(min, Math.min(max, steppedValue));

    if (value === undefined) {
      setInternalValue(clampedValue);
    }
    onChange?.(clampedValue);
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging]);

  const getTrackStyle = () => {
    switch (variant) {
      case 'rainbow':
        return {
          background: 'linear-gradient(90deg, #ff0096, #00ffff, #ffff00, #ff0096)',
          backgroundSize: '200% 100%',
          animation: 'gradient-shift 3s ease infinite'
        };
      case 'neon':
        const colors = {
          blue: '#3b82f6',
          red: '#ef4444',
          green: '#10b981',
          purple: '#8b5cf6',
          pink: '#ec4899',
          cyan: '#06b6d4'
        };
        return {
          background: colors[glowColor],
          boxShadow: `0 0 10px ${colors[glowColor]}40, 0 0 20px ${colors[glowColor]}20`
        };
      case 'gemini':
        return {
          background: 'linear-gradient(90deg, rgba(255, 0, 150, 0.8), rgba(0, 255, 255, 0.8), rgba(255, 255, 0, 0.8))',
          backgroundSize: '300% 100%',
          animation: 'gradient-shift 4s ease infinite'
        };
      default:
        return {
          background: '#3b82f6'
        };
    }
  };

  const getThumbStyle = () => {
    const baseStyle = {
      transform: `translateX(-50%) ${isDragging || isHovering ? 'scale(1.2)' : 'scale(1)'}`,
      transition: 'transform 0.2s ease'
    };

    switch (variant) {
      case 'rainbow':
        return {
          ...baseStyle,
          background: 'linear-gradient(135deg, #ff0096, #00ffff)',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3), 0 0 15px rgba(255, 0, 150, 0.5)'
        };
      case 'neon':
        const colors = {
          blue: '#3b82f6',
          red: '#ef4444',
          green: '#10b981',
          purple: '#8b5cf6',
          pink: '#ec4899',
          cyan: '#06b6d4'
        };
        return {
          ...baseStyle,
          background: colors[glowColor],
          boxShadow: `0 4px 12px rgba(0, 0, 0, 0.3), 0 0 15px ${colors[glowColor]}60`
        };
      case 'gemini':
        return {
          ...baseStyle,
          background: 'linear-gradient(135deg, rgba(255, 0, 150, 0.9), rgba(0, 255, 255, 0.9))',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3), 0 0 15px rgba(255, 0, 150, 0.4), 0 0 25px rgba(0, 255, 255, 0.2)'
        };
      default:
        return {
          ...baseStyle,
          background: 'white',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)'
        };
    }
  };

  return (
    <div className={cn('w-full space-y-3', className)}>
      {label && (
        <div className="flex justify-between items-center">
          <label className="text-sm font-medium text-gray-300">{label}</label>
          {showValue && (
            <span className="text-sm text-gray-400 font-mono">
              {currentValue.toFixed(step < 1 ? 2 : 0)}
            </span>
          )}
        </div>
      )}
      
      <div className="relative">
        {/* Track Background */}
        <div
          ref={sliderRef}
          className={cn(
            'relative w-full h-2 rounded-full cursor-pointer transition-all duration-300',
            disabled ? 'opacity-50 cursor-not-allowed' : 'hover:h-2.5'
          )}
          style={{
            background: 'rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(10px)'
          }}
          onMouseDown={handleMouseDown}
          onMouseEnter={() => setIsHovering(true)}
          onMouseLeave={() => setIsHovering(false)}
        >
          {/* Active Track */}
          <motion.div
            className="absolute top-0 left-0 h-full rounded-full"
            style={{
              width: `${percentage}%`,
              ...getTrackStyle()
            }}
            animate={{ width: `${percentage}%` }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          />
        </div>
        
        {/* Thumb */}
        <motion.div
          ref={thumbRef}
          className={cn(
            'absolute top-1/2 w-5 h-5 rounded-full cursor-pointer border-2 border-white/20',
            disabled ? 'cursor-not-allowed' : 'cursor-grab active:cursor-grabbing'
          )}
          style={{
            left: `${percentage}%`,
            ...getThumbStyle()
          }}
          animate={{ 
            left: `${percentage}%`,
            scale: isDragging ? 1.3 : isHovering ? 1.1 : 1
          }}
          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          whileHover={{ scale: disabled ? 1 : 1.1 }}
          whileTap={{ scale: disabled ? 1 : 1.3 }}
          onMouseDown={handleMouseDown}
          onMouseEnter={() => setIsHovering(true)}
          onMouseLeave={() => setIsHovering(false)}
        >
          {/* Inner glow effect */}
          <div 
            className="absolute inset-0 rounded-full opacity-50"
            style={{
              background: variant === 'rainbow' 
                ? 'linear-gradient(135deg, rgba(255, 0, 150, 0.3), rgba(0, 255, 255, 0.3))'
                : variant === 'gemini'
                ? 'linear-gradient(135deg, rgba(255, 0, 150, 0.2), rgba(0, 255, 255, 0.2))'
                : 'rgba(255, 255, 255, 0.2)'
            }}
          />
        </motion.div>

        {/* Value tooltip on hover/drag */}
        {(isHovering || isDragging) && (
          <motion.div
            className="absolute -top-10 px-2 py-1 text-xs font-medium text-white rounded-md pointer-events-none"
            style={{
              left: `${percentage}%`,
              transform: 'translateX(-50%)',
              background: 'rgba(0, 0, 0, 0.8)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 255, 255, 0.2)'
            }}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 5 }}
          >
            {currentValue.toFixed(step < 1 ? 1 : 0)}
          </motion.div>
        )}
      </div>
    </div>
  );
};