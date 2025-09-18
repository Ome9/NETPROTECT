import React from "react";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface FeatureCardProps {
  icon?: LucideIcon;
  iconColor?: string;
  title: string;
  description: string;
  className?: string;
  variant?: "default" | "gradient" | "glass" | "neon";
  onClick?: () => void;
}

export const FeatureCard: React.FC<FeatureCardProps> = ({
  icon: Icon,
  iconColor = "text-purple-500",
  title,
  description,
  className,
  variant = "default",
  onClick,
}) => {
  const variantStyles = {
    default:
      "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-lg hover:shadow-xl",
    gradient:
      "bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 border border-purple-200 dark:border-purple-800 shadow-lg hover:shadow-purple-500/25",
    glass:
      "bg-white/10 dark:bg-slate-900/10 backdrop-blur-lg border border-white/20 dark:border-slate-700/50 shadow-lg hover:shadow-xl",
    neon: "bg-slate-900 border-2 border-purple-500 shadow-lg shadow-purple-500/25 hover:shadow-purple-500/40 hover:border-purple-400",
  };

  const textStyles = {
    default: "text-slate-900 dark:text-slate-100",
    gradient: "text-slate-900 dark:text-slate-100",
    glass: "text-slate-900 dark:text-white",
    neon: "text-white",
  };

  const descriptionStyles = {
    default: "text-slate-600 dark:text-slate-400",
    gradient: "text-slate-600 dark:text-slate-400",
    glass: "text-slate-700 dark:text-slate-300",
    neon: "text-slate-300",
  };

  return (
    <div
      className={cn(
        "group relative rounded-2xl p-6 transition-all duration-300 hover:scale-105",
        variantStyles[variant],
        onClick && "cursor-pointer",
        className
      )}
      onClick={onClick}
    >
      {/* Background glow effect for neon variant */}
      {variant === "neon" && (
        <div className="absolute -inset-1 rounded-2xl bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 opacity-0 blur transition-opacity duration-300 group-hover:opacity-75" />
      )}

      <div className="relative">
        {/* Icon */}
        {Icon && (
          <div className="mb-4">
            <div
              className={cn(
                "inline-flex rounded-lg p-3",
                variant === "neon"
                  ? "bg-purple-500/20 text-purple-400"
                  : variant === "glass"
                  ? "bg-white/20 backdrop-blur-sm"
                  : "bg-purple-100 dark:bg-purple-900/30"
              )}
            >
              <Icon className={cn("h-6 w-6", iconColor)} />
            </div>
          </div>
        )}

        {/* Title */}
        <h3
          className={cn(
            "mb-2 text-xl font-semibold",
            textStyles[variant]
          )}
        >
          {title}
        </h3>

        {/* Description */}
        <p className={cn("leading-relaxed", descriptionStyles[variant])}>
          {description}
        </p>

        {/* Hover indicator */}
        <div className="absolute bottom-4 right-4 transform opacity-0 transition-all duration-300 group-hover:translate-x-1 group-hover:opacity-100">
          <div className="h-2 w-2 rounded-full bg-purple-500" />
        </div>
      </div>
    </div>
  );
};

interface FeatureGridProps {
  features: Array<Omit<FeatureCardProps, 'className'>>;
  columns?: 2 | 3 | 4;
  className?: string;
}

export const FeatureGrid: React.FC<FeatureGridProps> = ({
  features,
  columns = 3,
  className,
}) => {
  const gridCols = {
    2: "grid-cols-1 md:grid-cols-2",
    3: "grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
    4: "grid-cols-1 md:grid-cols-2 lg:grid-cols-4",
  };

  return (
    <div className={cn("grid gap-6", gridCols[columns], className)}>
      {features.map((feature, index) => (
        <FeatureCard
          key={index}
          {...feature}
          className="h-full"
        />
      ))}
    </div>
  );
};