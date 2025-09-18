import React, { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

interface TextRevealProps {
  text: string;
  className?: string;
  revealDuration?: number;
}

export const TextReveal: React.FC<TextRevealProps> = ({
  text,
  className,
  revealDuration = 2000,
}) => {
  const textRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (textRef.current) {
      const words = text.split(" ");
      textRef.current.innerHTML = words
        .map((word, index) => {
          return `<span class="inline-block opacity-0 transform translate-y-4" style="animation-delay: ${
            index * 0.1
          }s; animation: revealText 0.8s ease-out forwards;">${word}</span>`;
        })
        .join(" ");
    }
  }, [text]);

  return (
    <>
      <style jsx>{`
        @keyframes revealText {
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
      <div
        ref={textRef}
        className={cn("text-4xl font-bold text-white", className)}
      />
    </>
  );
};

interface TypingAnimationProps {
  text: string;
  duration?: number;
  className?: string;
}

export const TypingAnimation: React.FC<TypingAnimationProps> = ({
  text,
  duration = 200,
  className,
}) => {
  const [displayedText, setDisplayedText] = React.useState<string>("");
  const [i, setI] = React.useState<number>(0);

  React.useEffect(() => {
    const typingEffect = setInterval(() => {
      if (i < text.length) {
        setDisplayedText(text.substring(0, i + 1));
        setI(i + 1);
      } else {
        clearInterval(typingEffect);
      }
    }, duration);

    return () => {
      clearInterval(typingEffect);
    };
  }, [i]);

  return (
    <h1
      className={cn(
        "font-display text-center text-4xl font-bold leading-[5rem] tracking-[-0.02em] text-black drop-shadow-sm dark:text-white",
        className
      )}
    >
      {displayedText}
      <span className="animate-pulse">|</span>
    </h1>
  );
};