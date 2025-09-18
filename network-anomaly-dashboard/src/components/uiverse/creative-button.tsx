import React from "react";
import { cn } from "@/lib/utils";

interface CreativeButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: "neon" | "liquid" | "cyber" | "glass" | "rainbow" | "magnetic";
  size?: "sm" | "md" | "lg";
}

export const CreativeButton: React.FC<CreativeButtonProps> = ({
  children,
  variant = "neon",
  size = "md",
  className,
  ...props
}) => {
  const sizeStyles = {
    sm: "px-4 py-2 text-sm",
    md: "px-6 py-3 text-base",
    lg: "px-8 py-4 text-lg",
  };

  if (variant === "neon") {
    return (
      <button
        className={cn(
          "group relative overflow-hidden rounded-lg font-semibold transition-all duration-300",
          sizeStyles[size],
          className
        )}
        {...props}
      >
        <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600" />
        <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
        <div className="absolute inset-0.5 rounded-lg bg-slate-900" />
        <span className="relative z-10 text-white">{children}</span>
        <div className="absolute inset-0 rounded-lg opacity-0 shadow-lg shadow-purple-500/50 transition-opacity duration-300 group-hover:opacity-100" />
      </button>
    );
  }

  if (variant === "liquid") {
    return (
      <button
        className={cn(
          "group relative overflow-hidden rounded-full bg-gradient-to-r from-blue-500 to-purple-500 font-semibold text-white transition-all duration-500 hover:scale-105",
          sizeStyles[size],
          className
        )}
        {...props}
      >
        <span className="relative z-10">{children}</span>
        <div className="absolute inset-0 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 opacity-0 transition-opacity duration-500 group-hover:opacity-100" />
        <div className="absolute -inset-2 rounded-full bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 opacity-0 blur transition-opacity duration-500 group-hover:opacity-75" />
      </button>
    );
  }

  if (variant === "cyber") {
    return (
      <button
        className={cn(
          "group relative overflow-hidden border-2 border-cyan-400 bg-slate-900 font-mono font-bold text-cyan-400 transition-all duration-300 hover:bg-cyan-400 hover:text-slate-900",
          sizeStyles[size],
          className
        )}
        style={{
          clipPath: "polygon(0 0, calc(100% - 20px) 0, 100% 100%, 20px 100%)",
        }}
        {...props}
      >
        <span className="relative z-10">{children}</span>
        <div className="absolute inset-0 bg-cyan-400 translate-x-full transition-transform duration-300 group-hover:translate-x-0" />
      </button>
    );
  }

  if (variant === "glass") {
    return (
      <button
        className={cn(
          "group relative overflow-hidden rounded-2xl border border-white/20 bg-white/10 backdrop-blur-lg font-semibold text-white transition-all duration-300 hover:bg-white/20",
          sizeStyles[size],
          className
        )}
        {...props}
      >
        <span className="relative z-10">{children}</span>
        <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-white/10 to-white/5 opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
      </button>
    );
  }

  if (variant === "rainbow") {
    return (
      <button
        className={cn(
          "group relative overflow-hidden rounded-lg font-semibold text-white transition-all duration-300 hover:scale-105",
          sizeStyles[size],
          className
        )}
        {...props}
      >
        <div 
          className="absolute inset-0 rounded-lg bg-gradient-to-r from-red-500 via-yellow-500 via-green-500 via-blue-500 via-indigo-500 to-purple-500"
          style={{
            backgroundSize: "300% 300%",
            animation: "rainbow 3s ease infinite",
          }}
        />
        <div className="absolute inset-0.5 rounded-lg bg-slate-900" />
        <span className="relative z-10">{children}</span>
        <style jsx>{`
          @keyframes rainbow {
            0%, 100% {
              background-position: 0% 50%;
            }
            50% {
              background-position: 100% 50%;
            }
          }
        `}</style>
      </button>
    );
  }

  if (variant === "magnetic") {
    return (
      <button
        className={cn(
          "group relative rounded-lg bg-gradient-to-r from-purple-600 to-blue-600 font-semibold text-white transition-all duration-300",
          sizeStyles[size],
          className
        )}
        {...props}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left - rect.width / 2;
          const y = e.clientY - rect.top - rect.height / 2;
          e.currentTarget.style.transform = `translate(${x * 0.1}px, ${y * 0.1}px) scale(1.05)`;
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = "translate(0px, 0px) scale(1)";
        }}
      >
        <span className="relative z-10">{children}</span>
        <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
      </button>
    );
  }

  return (
    <button
      className={cn(
        "rounded-lg bg-blue-600 font-semibold text-white transition-colors hover:bg-blue-700",
        sizeStyles[size],
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
};