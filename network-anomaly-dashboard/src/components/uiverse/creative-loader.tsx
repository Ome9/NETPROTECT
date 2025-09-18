import React from "react";
import { cn } from "@/lib/utils";

interface CreativeLoaderProps {
  size?: "sm" | "md" | "lg";
  variant?: "dots" | "ripple" | "pulse" | "spinner" | "bars";
  color?: string;
  className?: string;
}

export const CreativeLoader: React.FC<CreativeLoaderProps> = ({
  size = "md",
  variant = "dots",
  color = "#8b5cf6",
  className,
}) => {
  const sizeStyles = {
    sm: "w-6 h-6",
    md: "w-12 h-12",
    lg: "w-24 h-24",
  };

  if (variant === "dots") {
    return (
      <div className={cn("flex items-center justify-center space-x-2", className)}>
        {[0, 1, 2].map((index) => (
          <div
            key={index}
            className={cn("rounded-full", size === "sm" ? "w-2 h-2" : size === "md" ? "w-3 h-3" : "w-4 h-4")}
            style={{
              backgroundColor: color,
              animation: `bounce 1.4s ease-in-out ${index * 0.16}s infinite both`,
            }}
          />
        ))}
        <style jsx>{`
          @keyframes bounce {
            0%, 80%, 100% {
              transform: scale(0.8);
              opacity: 0.5;
            }
            40% {
              transform: scale(1.2);
              opacity: 1;
            }
          }
        `}</style>
      </div>
    );
  }

  if (variant === "ripple") {
    return (
      <div className={cn("relative", sizeStyles[size], className)}>
        {[0, 1].map((index) => (
          <div
            key={index}
            className="absolute inset-0 rounded-full border-4 opacity-100"
            style={{
              borderColor: color,
              animation: `ripple 2s cubic-bezier(0, 0.2, 0.8, 1) ${index * 1}s infinite`,
            }}
          />
        ))}
        <style jsx>{`
          @keyframes ripple {
            0% {
              transform: scale(0);
              opacity: 1;
            }
            100% {
              transform: scale(1);
              opacity: 0;
            }
          }
        `}</style>
      </div>
    );
  }

  if (variant === "pulse") {
    return (
      <div
        className={cn("rounded-full", sizeStyles[size], className)}
        style={{
          backgroundColor: color,
          animation: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        }}
      >
        <style jsx>{`
          @keyframes pulse {
            0%, 100% {
              opacity: 1;
            }
            50% {
              opacity: 0.5;
            }
          }
        `}</style>
      </div>
    );
  }

  if (variant === "spinner") {
    return (
      <div
        className={cn("rounded-full border-4 border-gray-200", sizeStyles[size], className)}
        style={{
          borderTopColor: color,
          animation: "spin 1s linear infinite",
        }}
      >
        <style jsx>{`
          @keyframes spin {
            to {
              transform: rotate(360deg);
            }
          }
        `}</style>
      </div>
    );
  }

  if (variant === "bars") {
    return (
      <div className={cn("flex items-end space-x-1", className)}>
        {[0, 1, 2, 3, 4].map((index) => (
          <div
            key={index}
            className={cn(
              "rounded-sm",
              size === "sm" ? "w-1 h-4" : size === "md" ? "w-2 h-8" : "w-3 h-12"
            )}
            style={{
              backgroundColor: color,
              animation: `bars 1.2s ease-in-out ${index * 0.1}s infinite`,
            }}
          />
        ))}
        <style jsx>{`
          @keyframes bars {
            0%, 40%, 100% {
              transform: scaleY(0.4);
            }
            20% {
              transform: scaleY(1);
            }
          }
        `}</style>
      </div>
    );
  }

  return null;
};

interface LoadingOverlayProps {
  loading: boolean;
  children: React.ReactNode;
  loaderProps?: CreativeLoaderProps;
  message?: string;
  className?: string;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  loading,
  children,
  loaderProps,
  message,
  className,
}) => {
  return (
    <div className={cn("relative", className)}>
      {children}
      {loading && (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black/50 backdrop-blur-sm">
          <CreativeLoader {...loaderProps} />
          {message && (
            <p className="mt-4 text-sm text-white">{message}</p>
          )}
        </div>
      )}
    </div>
  );
};