import React from "react";
import { cn } from "@/lib/utils";

interface ModernHeroProps {
  title: string;
  subtitle: string;
  description?: string;
  primaryButton?: {
    text: string;
    onClick?: () => void;
  };
  secondaryButton?: {
    text: string;
    onClick?: () => void;
  };
  className?: string;
  backgroundImage?: string;
}

export const ModernHero: React.FC<ModernHeroProps> = ({
  title,
  subtitle,
  description,
  primaryButton,
  secondaryButton,
  className,
  backgroundImage,
}) => {
  return (
    <section
      className={cn(
        "relative overflow-hidden bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white",
        className
      )}
      style={{
        backgroundImage: backgroundImage
          ? `linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url(${backgroundImage})`
          : undefined,
        backgroundSize: "cover",
        backgroundPosition: "center",
      }}
    >
      {/* Animated background pattern */}
      <div className="absolute inset-0 opacity-20">
        <svg
          className="absolute h-full w-full"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
        >
          <defs>
            <pattern
              id="hero-pattern"
              x="0"
              y="0"
              width="20"
              height="20"
              patternUnits="userSpaceOnUse"
            >
              <circle cx="10" cy="10" r="1" fill="currentColor" />
            </pattern>
          </defs>
          <rect width="100" height="100" fill="url(#hero-pattern)" />
        </svg>
      </div>

      <div className="relative mx-auto max-w-7xl px-4 py-24 sm:px-6 lg:px-8">
        <div className="text-center">
          {/* Subtitle */}
          <p className="mb-4 text-lg font-semibold text-purple-300">
            {subtitle}
          </p>

          {/* Main Title */}
          <h1 className="mb-6 text-4xl font-bold tracking-tight sm:text-6xl lg:text-7xl">
            <span className="bg-gradient-to-r from-purple-400 via-pink-500 to-purple-400 bg-clip-text text-transparent">
              {title}
            </span>
          </h1>

          {/* Description */}
          {description && (
            <p className="mx-auto mb-8 max-w-3xl text-xl text-slate-300">
              {description}
            </p>
          )}

          {/* Action Buttons */}
          <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
            {primaryButton && (
              <button
                onClick={primaryButton.onClick}
                className="group relative inline-flex items-center justify-center overflow-hidden rounded-full bg-gradient-to-r from-purple-500 to-pink-500 px-8 py-3 text-lg font-semibold text-white transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-purple-500/25"
              >
                <span className="relative z-10">{primaryButton.text}</span>
                <div className="absolute inset-0 -z-10 rounded-full bg-gradient-to-r from-purple-600 to-pink-600 opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
              </button>
            )}

            {secondaryButton && (
              <button
                onClick={secondaryButton.onClick}
                className="inline-flex items-center justify-center rounded-full border-2 border-white/20 px-8 py-3 text-lg font-semibold text-white transition-all duration-300 hover:border-white/40 hover:bg-white/10"
              >
                {secondaryButton.text}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Bottom gradient fade */}
      <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-slate-900 to-transparent" />
    </section>
  );
};