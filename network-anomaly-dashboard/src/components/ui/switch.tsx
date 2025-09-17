'use client';

import * as React from 'react';

interface SwitchProps {
  checked?: boolean;
  onCheckedChange?: (checked: boolean) => void;
  disabled?: boolean;
  className?: string;
}

const Switch = React.forwardRef<HTMLButtonElement, SwitchProps>(
  ({ checked = false, onCheckedChange, disabled = false, className = '', ...props }, ref) => {
    return (
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        className={`
          peer inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full 
          border-2 border-transparent transition-colors 
          focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 
          focus-visible:ring-offset-2 focus-visible:ring-offset-black 
          disabled:cursor-not-allowed disabled:opacity-50 
          ${checked 
            ? 'bg-gradient-to-r from-blue-500 to-purple-600' 
            : 'bg-gray-700'
          } 
          ${className}
        `}
        onClick={() => !disabled && onCheckedChange?.(!checked)}
        ref={ref}
        {...props}
      >
        <div
          className={`
            pointer-events-none block h-5 w-5 rounded-full bg-white shadow-lg 
            ring-0 transition-transform 
            ${checked ? 'translate-x-5' : 'translate-x-0'}
          `}
        />
      </button>
    );
  }
);

Switch.displayName = 'Switch';

export { Switch };