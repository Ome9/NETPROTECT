import React from "react";
import { cn } from "@/lib/utils";

interface GlowButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  glowColor?: string;
  borderColor?: string;
}

export const GlowButton: React.FC<GlowButtonProps> = ({
  children,
  className,
  glowColor = "rgb(59, 130, 246)",
  borderColor = "rgb(59, 130, 246)",
  ...props
}) => {
  return (
    <button
      className={cn(
        "relative inline-flex h-12 overflow-hidden rounded-full p-[1px] focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 focus:ring-offset-slate-50 transition-all duration-300",
        className
      )}
      {...props}
    >
      <span
        className="absolute inset-[-1000%] animate-[spin_2s_linear_infinite]"
        style={{
          background: `conic-gradient(from 90deg at 50% 50%, transparent 0%, ${borderColor} 50%, transparent 100%)`,
        }}
      />
      <span
        className="inline-flex h-full w-full cursor-pointer items-center justify-center rounded-full bg-slate-950 px-6 py-1 text-sm font-medium text-white backdrop-blur-3xl transition-all duration-300 hover:bg-slate-800"
        style={{
          boxShadow: `0 0 20px ${glowColor}40`,
        }}
      >
        {children}
      </span>
    </button>
  );
};

interface MovingBorderButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  borderRadius?: string;
  containerClassName?: string;
  borderClassName?: string;
  duration?: number;
}

export const MovingBorderButton: React.FC<MovingBorderButtonProps> = ({
  children,
  borderRadius = "1.75rem",
  containerClassName,
  borderClassName,
  duration = 2000,
  className,
  ...props
}) => {
  return (
    <button
      className={cn(
        "relative inline-flex h-12 overflow-hidden p-[1px] focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 focus:ring-offset-slate-50",
        containerClassName
      )}
      style={{ borderRadius }}
      {...props}
    >
      <div
        className={cn(
          "absolute inset-[-1000%]",
          "animate-[spin_2s_linear_infinite]",
          "bg-[conic-gradient(from_90deg_at_50%_50%,#E2CBFF_0%,#393BB2_50%,#E2CBFF_100%)]",
          borderClassName
        )}
        style={{
          animation: `spin ${duration / 1000}s linear infinite`,
        }}
      />
      <span
        className={cn(
          "inline-flex h-full w-full cursor-pointer items-center justify-center backdrop-blur-3xl",
          "bg-slate-950 px-6 py-1 text-sm font-medium text-white",
          "rounded-full transition-colors duration-300 hover:bg-slate-800",
          className
        )}
        style={{ borderRadius }}
      >
        {children}
      </span>
    </button>
  );
};