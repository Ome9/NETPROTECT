'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { AlertTriangle, Shield, Activity, Zap, Eye, Target, Router, Server, Monitor, HelpCircle } from 'lucide-react';

interface NetworkNode {
  id: string;
  ip: string;
  type: 'router' | 'server' | 'workstation' | 'unknown';
  status: 'normal' | 'suspicious' | 'threat';
  connections: number;
  lastSeen: number;
  riskScore: number;
}

interface NetworkTopologyProps {
  nodes: NetworkNode[];
  onNodeClick: (node: NetworkNode) => void;
}

export const NetworkTopologyVisualizer: React.FC<NetworkTopologyProps> = ({ 
  nodes = [], 
  onNodeClick 
}) => {
  const [selectedNode, setSelectedNode] = useState<NetworkNode | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'topology'>('topology');
  
  // Generate mock nodes if none provided
  const mockNodes: NetworkNode[] = [
    { id: '1', ip: '192.168.1.1', type: 'router', status: 'normal', connections: 45, lastSeen: Date.now() - 1000, riskScore: 0.1 },
    { id: '2', ip: '192.168.1.100', type: 'server', status: 'suspicious', connections: 23, lastSeen: Date.now() - 5000, riskScore: 0.7 },
    { id: '3', ip: '192.168.1.150', type: 'workstation', status: 'threat', connections: 12, lastSeen: Date.now() - 2000, riskScore: 0.9 },
    { id: '4', ip: '192.168.1.200', type: 'workstation', status: 'normal', connections: 8, lastSeen: Date.now() - 3000, riskScore: 0.2 },
    { id: '5', ip: '10.0.0.50', type: 'server', status: 'suspicious', connections: 67, lastSeen: Date.now() - 1500, riskScore: 0.6 },
    { id: '6', ip: '172.16.0.10', type: 'unknown', status: 'threat', connections: 89, lastSeen: Date.now() - 500, riskScore: 0.95 },
  ];

  const displayNodes = nodes.length > 0 ? nodes : mockNodes;

  const getNodeColor = (status: string) => {
    switch (status) {
      case 'threat': return 'bg-red-500 neon-glow-red';
      case 'suspicious': return 'bg-yellow-500';
      default: return 'bg-green-500 neon-glow-green';
    }
  };

  const getNodeIcon = (type: string) => {
    switch (type) {
      case 'router': return <Router className="h-4 w-4" />;
      case 'server': return <Server className="h-4 w-4" />;
      case 'workstation': return <Monitor className="h-4 w-4" />;
      default: return <HelpCircle className="h-4 w-4" />;
    }
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'threat': return 'pulse';
      case 'suspicious': return 'neon';
      default: return 'glass';
    }
  };



  const handleNodeClick = (node: NetworkNode) => {
    setSelectedNode(node);
    onNodeClick?.(node);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card variant="rainbow" className="shadow-2xl">
        <CardHeader>
          <div className="flex justify-between items-center">
            <CardTitle className="text-white flex items-center gap-2">
              <Shield className="h-5 w-5 text-blue-400" />
              Network Topology
            </CardTitle>
            <div className="flex gap-2">
              <Button 
                variant={viewMode === 'topology' ? 'rainbow' : 'glass'}
                size="sm"
                onClick={() => setViewMode('topology')}
                className="text-xs"
              >
                Topology
              </Button>
              <Button 
                variant={viewMode === 'grid' ? 'rainbow' : 'glass'}
                size="sm"
                onClick={() => setViewMode('grid')}
                className="text-xs"
              >
                Grid
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {viewMode === 'topology' ? (
            <div className="relative h-80 cyber-grid rounded-lg overflow-hidden">
              {/* SVG Definitions */}
              <svg className="absolute inset-0 w-full h-full pointer-events-none">
                <defs>
                  <linearGradient id="rainbow-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#ff0000" />
                    <stop offset="16.66%" stopColor="#ff7f00" />
                    <stop offset="33.33%" stopColor="#ffff00" />
                    <stop offset="50%" stopColor="#00ff00" />
                    <stop offset="66.66%" stopColor="#0000ff" />
                    <stop offset="83.33%" stopColor="#4b0082" />
                    <stop offset="100%" stopColor="#9400d3" />
                  </linearGradient>
                </defs>
              </svg>
              
              {/* Topology View */}
              <div className="absolute inset-0 p-4">
                {displayNodes.map((node, index) => {
                  const x = (index % 3) * 33.33 + 16.67;
                  const y = Math.floor(index / 3) * 25 + 12.5;
                  
                  return (
                    <motion.div
                      key={node.id}
                      className="absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer group"
                      style={{ left: `${x}%`, top: `${y}%` }}
                      onClick={() => handleNodeClick(node)}
                      whileHover={{ scale: 1.2 }}
                      whileTap={{ scale: 0.9 }}
                      animate={node.status === 'threat' ? { 
                        scale: [1, 1.1, 1],
                        boxShadow: [
                          "0 0 0px rgba(239, 68, 68, 0)",
                          "0 0 20px rgba(239, 68, 68, 0.8)",
                          "0 0 0px rgba(239, 68, 68, 0)"
                        ]
                      } : {}}
                      transition={{ 
                        duration: 2, 
                        repeat: node.status === 'threat' ? Infinity : 0 
                      }}
                    >
                      {/* Connection lines */}
                      {index > 0 && (
                        <svg className="absolute inset-0 w-full h-full pointer-events-none">
                          <line
                            x1="50%"
                            y1="50%"
                            x2={`${((index - 1) % 3) * 33.33 + 16.67 - x + 50}%`}
                            y2={`${Math.floor((index - 1) / 3) * 25 + 12.5 - y + 50}%`}
                            stroke="url(#rainbow-gradient)"
                            strokeWidth="2"
                            opacity="0.6"
                          />
                        </svg>
                      )}
                      
                      {/* Node */}
                      <Tooltip>
                        <TooltipTrigger>
                          <div className={`relative w-12 h-12 rounded-full ${getNodeColor(node.status)} 
                                         flex items-center justify-center text-white shadow-lg
                                         transition-all duration-300
                                         ${selectedNode?.id === node.id ? 'ring-2 ring-blue-400' : ''}`}>
                            {getNodeIcon(node.type)}
                          </div>
                        </TooltipTrigger>
                        <TooltipContent>
                          <div className="text-sm">
                            <p className="font-semibold">{node.ip}</p>
                            <p>Type: {node.type}</p>
                            <p>Status: {node.status}</p>
                            <p>Risk Score: {(node.riskScore * 100).toFixed(1)}%</p>
                            <p>Connections: {node.connections}</p>
                          </div>
                        </TooltipContent>
                      </Tooltip>
                      
                      {/* Risk indicator */}
                      {node.riskScore > 0.5 && (
                        <div className="absolute -top-1 -right-1">
                          <AlertTriangle className="h-4 w-4 text-red-400 animate-pulse" />
                        </div>
                      )}
                      
                      {/* Node label */}
                      <div className="absolute top-14 left-1/2 transform -translate-x-1/2 
                                     text-xs text-gray-300 whitespace-nowrap opacity-0 
                                     group-hover:opacity-100 transition-opacity">
                        {node.ip}
                      </div>
                    </motion.div>
                );
              })}
            </div>
            
            {/* Legend */}
            <div className="absolute bottom-2 left-2 text-xs text-gray-400 space-y-1">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>Normal</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                <span>Suspicious</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                <span>Threat</span>
              </div>
            </div>
          </div>
        ) : (
          // Enhanced Grid View
          <div className="grid grid-cols-2 gap-3 max-h-80 overflow-y-auto">
            {displayNodes.map((node, index) => (
              <motion.div
                key={node.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                whileHover={{ scale: 1.02 }}
                className={`p-3 rounded-lg border cursor-pointer transition-all duration-200 glass-effect
                           ${selectedNode?.id === node.id 
                             ? 'border-blue-400 shadow-lg shadow-blue-400/20' 
                             : 'border-gray-600 hover:border-gray-500'}`}
                onClick={() => handleNodeClick(node)}
              >
                <div className="flex items-center gap-2 mb-2">
                  {getNodeIcon(node.type)}
                  <span className="text-sm font-medium text-white">{node.ip}</span>
                  <Badge 
                    variant={getStatusBadgeVariant(node.status)}
                    className="text-xs"
                  >
                    {node.status}
                  </Badge>
                </div>
                <div className="text-xs text-gray-300 space-y-1">
                  <div>Type: <span className="text-blue-400">{node.type}</span></div>
                  <div>Connections: <span className="text-green-400">{node.connections}</span></div>
                  <div>Risk: <span className={`font-semibold ${
                    node.riskScore > 0.7 ? 'text-red-400' :
                    node.riskScore > 0.4 ? 'text-yellow-400' : 'text-green-400'
                  }`}>{(node.riskScore * 100).toFixed(1)}%</span></div>
                </div>
                <Progress 
                  value={node.riskScore * 100} 
                  variant={node.riskScore > 0.7 ? 'neon' : 'rainbow'} 
                  glowColor={node.riskScore > 0.7 ? 'red' : 'green'}
                  className="mt-2 h-1" 
                />
              </motion.div>
            ))}
          </div>
        )}
        
        {/* Enhanced Selected Node Details */}
        <AnimatePresence>
          {selectedNode && (
            <motion.div 
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 p-4 glass-effect rounded-lg border border-gray-600"
            >
              <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                {getNodeIcon(selectedNode.type)}
                Node Details - {selectedNode.ip}
              </h4>
              <div className="grid grid-cols-2 gap-4 text-xs text-gray-300">
                <div className="space-y-2">
                  <div>
                    <span className="text-gray-400">IP Address:</span> 
                    <span className="ml-2 text-blue-400 font-mono">{selectedNode.ip}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Type:</span> 
                    <span className="ml-2 text-green-400 capitalize">{selectedNode.type}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Status:</span> 
                    <Badge variant={getStatusBadgeVariant(selectedNode.status)} className="text-xs ml-2">
                      {selectedNode.status}
                    </Badge>
                  </div>
                </div>
                <div className="space-y-2">
                  <div>
                    <span className="text-gray-400">Risk Score:</span> 
                    <span className={`ml-2 font-semibold ${
                      selectedNode.riskScore > 0.7 ? 'text-red-400' :
                      selectedNode.riskScore > 0.4 ? 'text-yellow-400' : 'text-green-400'
                    }`}>
                      {(selectedNode.riskScore * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Connections:</span> 
                    <span className="ml-2 text-purple-400">{selectedNode.connections}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Last Seen:</span> 
                    <span className="ml-2 text-cyan-400">{new Date(selectedNode.lastSeen).toLocaleTimeString()}</span>
                  </div>
                </div>
              </div>
              <div className="mt-3">
                <span className="text-gray-400 text-xs">Risk Assessment:</span>
                <Progress 
                  value={selectedNode.riskScore * 100} 
                  variant={selectedNode.riskScore > 0.7 ? 'neon' : 'rainbow'} 
                  glowColor={selectedNode.riskScore > 0.7 ? 'red' : 'green'}
                  className="mt-1 h-2" 
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        </CardContent>
      </Card>
    </motion.div>
  );
};