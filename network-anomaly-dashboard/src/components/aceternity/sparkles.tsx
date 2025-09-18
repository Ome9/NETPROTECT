import React, { useId, useState, useEffect } from "react";
import { cn } from "@/lib/utils";

interface SparklesProps {
  id?: string;
  background?: string;
  particleColor?: string;
  className?: string;
  particleSize?: number;
  particleDensity?: number;
}

// Seeded random function for consistent results
function seededRandom(seed: number): number {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

// Generate deterministic sparkle data
function generateSparkleData(particleDensity: number, particleSize: number, seed: string) {
  const seedNum = seed.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  
  return Array.from({ length: particleDensity }, (_, index) => {
    const baseSeed = seedNum + index;
    return {
      id: index,
      cx: seededRandom(baseSeed * 1.1) * 400,
      cy: seededRandom(baseSeed * 1.3) * 400,
      opacity: seededRandom(baseSeed * 1.7) * 0.8 + 0.2,
      opacityDuration: seededRandom(baseSeed * 2.1) * 3 + 1,
      radiusDuration: seededRandom(baseSeed * 2.3) * 2 + 1,
    };
  });
}

export const SparklesCore: React.FC<SparklesProps> = (props) => {
  const {
    id,
    className,
    background = "transparent",
    particleSize = 1.2,
    particleColor = "#ffffff",
    particleDensity = 100,
  } = props;

  const generatedId = useId();
  const sparkleId = id || generatedId;
  
  const [mounted, setMounted] = useState(false);
  const [sparkleData, setSparkleData] = useState<ReturnType<typeof generateSparkleData>>([]);

  useEffect(() => {
    setMounted(true);
    setSparkleData(generateSparkleData(particleDensity, particleSize, sparkleId));
  }, [particleDensity, particleSize, sparkleId]);

  // Don't render sparkles on server to avoid hydration mismatch
  if (!mounted) {
    return (
      <div className={cn("h-full w-full", className)}>
        <svg
          className="h-full w-full"
          viewBox="0 0 400 400"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <filter id={`sparkle-filter-${sparkleId}`}>
              <feGaussianBlur
                stdDeviation="3"
                result="coloredBlur"
              />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
        </svg>
      </div>
    );
  }

  return (
    <div className={cn("h-full w-full", className)}>
      <svg
        className="h-full w-full"
        viewBox="0 0 400 400"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <filter id={`sparkle-filter-${sparkleId}`}>
            <feGaussianBlur
              stdDeviation="3"
              result="coloredBlur"
            />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        {sparkleData.map((sparkle) => (
          <circle
            key={sparkle.id}
            cx={sparkle.cx}
            cy={sparkle.cy}
            r={particleSize}
            fill={particleColor}
            filter={`url(#sparkle-filter-${sparkleId})`}
            opacity={sparkle.opacity}
          >
            <animate
              attributeName="opacity"
              values="0.2;1;0.2"
              dur={sparkle.opacityDuration + "s"}
              repeatCount="indefinite"
            />
            <animate
              attributeName="r"
              values={`${particleSize * 0.5};${particleSize * 1.5};${particleSize * 0.5}`}
              dur={sparkle.radiusDuration + "s"}
              repeatCount="indefinite"
            />
          </circle>
        ))}
      </svg>
    </div>
  );
};

export const Sparkles: React.FC<
  SparklesProps & { children: React.ReactNode }
> = ({ children, className, ...props }) => {
  return (
    <div className={cn("relative", className)}>
      <div className="absolute inset-0 z-0">
        <SparklesCore {...props} />
      </div>
      <div className="relative z-10">{children}</div>
    </div>
  );
};