import * as React from "react"
import { cn } from "@/lib/utils"

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'rainbow' | 'glass' | 'neon' | 'gemini' | 'selected';
  glowColor?: 'blue' | 'red' | 'green' | 'purple';
  isSelected?: boolean;
  onSelect?: () => void;
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = 'default', glowColor = 'blue', isSelected = false, onSelect, children, ...props }, ref) => {
    const getVariantClasses = () => {
      switch (variant) {
        case 'rainbow':
          return "rainbow-border rainbow-shadow bg-card/90 backdrop-blur-sm";
        case 'glass':
          return "glass-effect border-white/10";
        case 'neon':
          return `bg-card border-${glowColor}-500/50 neon-glow-${glowColor}`;
        case 'gemini':
          return "card-gemini";
        case 'selected':
          return "gradient-border-active";
        default:
          return "border bg-card";
      }
    };

    const handleClick = () => {
      if (onSelect) {
        onSelect();
      }
    };

    // Enhanced selected state with pop-out effect and background blur
    if (variant === 'selected' || isSelected) {
      return (
        <>
          {/* Background blur overlay */}
          <div className="blur-overlay active" />
          
          {/* Selected card with gradient border */}
          <div
            ref={ref}
            className={cn(
              "relative rounded-2xl text-card-foreground cursor-pointer z-50",
              "transform scale-105 translate-y-[-8px]",
              "transition-all duration-500 ease-out",
              "shadow-2xl",
              className
            )}
            onClick={handleClick}
            style={{
              background: 'linear-gradient(45deg, #ff0096, #00ffff, #ffff00, #ff0096)',
              backgroundSize: '400% 400%',
              animation: 'gradient-shift 3s ease infinite',
              padding: '3px',
              boxShadow: `
                0 25px 50px -12px rgba(0, 0, 0, 0.6),
                0 0 40px rgba(255, 0, 150, 0.4),
                0 0 80px rgba(0, 255, 255, 0.3),
                0 0 120px rgba(255, 255, 0, 0.2)
              `
            }}
            {...props}
          >
            {/* Inner content container */}
            <div 
              className="relative bg-card rounded-xl backdrop-blur-xl border-0 overflow-hidden"
              style={{
                background: 'rgba(26, 26, 26, 0.95)',
                backdropFilter: 'blur(20px)'
              }}
            >
              {children}
            </div>
          </div>
        </>
      );
    }

    // Gemini-style hover effect
    if (variant === 'gemini') {
      return (
        <div
          ref={ref}
          className={cn(
            "relative rounded-2xl text-card-foreground transition-all duration-500 ease-out group",
            "bg-card/80 backdrop-blur-xl border border-white/10",
            "hover:transform hover:translate-y-[-4px]",
            "hover:shadow-2xl hover:border-pink-500/30",
            onSelect && "cursor-pointer",
            className
          )}
          onClick={handleClick}
          {...props}
        >
          {/* Gradient border that appears on hover */}
          <div 
            className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
            style={{
              background: 'linear-gradient(135deg, rgba(255, 0, 150, 0.2), rgba(0, 255, 255, 0.2), rgba(255, 255, 0, 0.2))',
              backgroundSize: '400% 400%',
              animation: 'gradient-shift 4s ease infinite'
            }}
          />
          
          {/* Subtle glow effect */}
          <div 
            className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
            style={{
              boxShadow: '0 0 30px rgba(255, 0, 150, 0.2), 0 0 60px rgba(0, 255, 255, 0.1)'
            }}
          />
          
          {/* Content */}
          <div className="relative z-10">
            {children}
          </div>
        </div>
      );
    }

    return (
      <div
        ref={ref}
        className={cn(
          "rounded-lg text-card-foreground shadow-sm transition-all duration-300 hover:shadow-lg",
          getVariantClasses(),
          onSelect && "cursor-pointer",
          className
        )}
        onClick={handleClick}
        {...props}
      >
        {children}
      </div>
    );
  }
);
Card.displayName = "Card"

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "text-2xl font-semibold leading-none tracking-tight",
      className
    )}
    {...props}
  />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
    {...props}
  />
))
CardFooter.displayName = "CardFooter"

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent }