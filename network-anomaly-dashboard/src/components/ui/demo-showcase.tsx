'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { EnhancedSlider } from '@/components/ui/enhanced-slider';
import { Fab } from '@/components/ui/fab';
import { 
  Sparkles, Zap, Heart, Star, Rocket, 
  Shield, Brain, Activity, TrendingUp, Settings 
} from 'lucide-react';

export const DemoShowcase: React.FC = () => {
  const [selectedCard, setSelectedCard] = useState<string | null>(null);
  const [sliderValue, setSliderValue] = useState(50);
  const [progressValue, setProgressValue] = useState(75);

  const demoCards = [
    {
      id: 'gemini-1',
      title: 'Gemini AI Style',
      description: 'Hover to see the gradient border effect',
      icon: <Sparkles className="h-5 w-5" />,
      variant: 'gemini' as const,
      value: 92.5
    },
    {
      id: 'selected-1',
      title: 'Selected State',
      description: 'Click to see pop-out effect with blur',
      icon: <Star className="h-5 w-5" />,
      variant: 'default' as const,
      value: 88.3
    },
    {
      id: 'rainbow-1',
      title: 'Rainbow Gradient',
      description: 'Animated rainbow border',
      icon: <Rocket className="h-5 w-5" />,
      variant: 'rainbow' as const,
      value: 95.7
    }
  ];

  return (
    <div className="space-y-8 p-6">
      {/* Header */}
      <div className="text-center space-y-4">
        <motion.h1 
          className="text-4xl font-bold text-white"
          animate={{ 
            backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"] 
          }}
          transition={{ 
            duration: 3, 
            repeat: Infinity, 
            ease: "linear" 
          }}
        >
          Enhanced UI Showcase
        </motion.h1>
        <p className="text-gray-300 text-lg">
          Experience the next-level UI with Gemini AI-inspired gradients and animations
        </p>
      </div>

      {/* Card Variants Demo */}
      <div className="space-y-4">
        <h2 className="text-2xl font-semibold text-white mb-4">Card Variants</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {demoCards.map((card, index) => (
            <motion.div
              key={card.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card 
                variant={selectedCard === card.id ? 'selected' : card.variant}
                isSelected={selectedCard === card.id}
                onSelect={() => setSelectedCard(selectedCard === card.id ? null : card.id)}
                className="h-full cursor-pointer"
              >
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    {card.icon}
                    {card.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-300 mb-4">{card.description}</p>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-400">Performance</span>
                      <Badge variant="rainbow" className="text-xs">
                        {card.value}%
                      </Badge>
                    </div>
                    <Progress value={card.value} variant="rainbow" className="h-2" />
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Interactive Components */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Sliders Demo */}
        <Card variant="gemini">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Interactive Sliders
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <EnhancedSlider
              label="Rainbow Slider"
              variant="rainbow"
              value={sliderValue}
              onChange={setSliderValue}
              min={0}
              max={100}
            />
            <EnhancedSlider
              label="Neon Blue Slider"
              variant="neon"
              glowColor="blue"
              value={progressValue}
              onChange={setProgressValue}
              min={0}
              max={100}
            />
            <EnhancedSlider
              label="Gemini Slider"
              variant="gemini"
              defaultValue={25}
              min={0}
              max={100}
            />
          </CardContent>
        </Card>

        {/* Progress Bars Demo */}
        <Card variant="gemini">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Progress Indicators
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm text-gray-300">Rainbow Progress</span>
                <span className="text-sm text-gray-400">85%</span>
              </div>
              <Progress value={85} variant="rainbow" className="h-3" />
            </div>
            
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm text-gray-300">Neon Green</span>
                <span className="text-sm text-gray-400">92%</span>
              </div>
              <Progress value={92} variant="neon" glowColor="green" className="h-3" />
            </div>
            
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm text-gray-300">Neon Purple</span>
                <span className="text-sm text-gray-400">67%</span>
              </div>
              <Progress value={67} variant="neon" glowColor="purple" className="h-3" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Button Variants */}
      <Card variant="gemini">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Enhanced Buttons
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4">
            <Button variant="rainbow" size="lg">
              <Sparkles className="h-4 w-4 mr-2" />
              Rainbow Button
            </Button>
            <Button variant="neon" size="lg">
              <Zap className="h-4 w-4 mr-2" />
              Neon Button
            </Button>
            <Button variant="glass" size="lg">
              <Shield className="h-4 w-4 mr-2" />
              Glass Button
            </Button>
            <Button variant="default" size="lg">
              <Brain className="h-4 w-4 mr-2" />
              Default Button
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Badge Variants */}
      <Card variant="gemini">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Star className="h-5 w-5" />
            Enhanced Badges
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            <Badge variant="rainbow">Rainbow Badge</Badge>
            <Badge variant="neon">Neon Badge</Badge>
            <Badge variant="glass">Glass Badge</Badge>
            <Badge variant="pulse">Pulse Badge</Badge>
            <Badge variant="default">Default Badge</Badge>
          </div>
        </CardContent>
      </Card>

      {/* Floating Action Buttons */}
      <div className="relative h-32 border border-white/10 rounded-xl bg-black/20 backdrop-blur-sm">
        <div className="absolute inset-0 flex items-center justify-center text-gray-400">
          <p>Floating Action Buttons Demo Area</p>
        </div>
        <Fab 
          icon={<Heart className="h-6 w-6" />}
          variant="rainbow"
          position="bottom-right"
          tooltip="Rainbow FAB"
        />
        <Fab 
          icon="settings"
          variant="neon"
          position="bottom-left"
          tooltip="Settings FAB"
        />
      </div>
    </div>
  );
};