'use client';

import dynamic from 'next/dynamic';
import React from 'react';

// Wrapper to suppress hydration warnings for animated components
export function NoSSR({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  return <>{children}</>;
}

// Dynamic imports for components that might cause hydration issues
export const SparklesCore = dynamic(
  () => import('./aceternity/sparkles').then(mod => ({ default: mod.SparklesCore })),
  { 
    ssr: false,
    loading: () => <div className="h-full w-full" />
  }
);

export const AnimatedBackground = dynamic(
  () => import('./ui/animated-background').then(mod => ({ default: mod.AnimatedBackground })),
  { 
    ssr: false,
    loading: () => <div className="min-h-screen" />
  }
);