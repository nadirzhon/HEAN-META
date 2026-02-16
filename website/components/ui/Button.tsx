"use client";

import { useRef } from "react";
import clsx from "clsx";
import { type HTMLMotionProps, motion, useMotionValue, useSpring } from "framer-motion";

type ButtonVariant = "primary" | "outline";

interface ButtonProps extends Omit<HTMLMotionProps<"button">, "children"> {
  variant?: ButtonVariant;
  children: React.ReactNode;
}

export default function Button({
  variant = "primary",
  children,
  className,
  ...props
}: ButtonProps) {
  const buttonRef = useRef<HTMLButtonElement>(null);
  const x = useSpring(useMotionValue(0), { stiffness: 300, damping: 20 });
  const y = useSpring(useMotionValue(0), { stiffness: 300, damping: 20 });

  const handleMouseMove = (e: React.MouseEvent<HTMLButtonElement>) => {
    if (!buttonRef.current) return;
    const rect = buttonRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const deltaX = (e.clientX - centerX) * 0.08;
    const deltaY = (e.clientY - centerY) * 0.08;
    x.set(Math.max(-3, Math.min(3, deltaX)));
    y.set(Math.max(-3, Math.min(3, deltaY)));
  };

  const handleMouseLeave = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <motion.button
      ref={buttonRef}
      whileHover={{ scale: 1.04 }}
      whileTap={{ scale: 0.97 }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{ x, y }}
      className={clsx(
        "relative cursor-pointer rounded-full px-6 py-3 sm:px-8 sm:py-4 text-sm font-semibold tracking-wide transition-all duration-300 min-h-[44px]",
        variant === "primary" && [
          "bg-[#8F45E6] text-supernova glow-axon",
          "hover:bg-[#8340DD] hover:shadow-[0_0_32px_rgba(164,91,255,0.5)]",
        ],
        variant === "outline" && [
          "border border-stream/50 text-stream bg-transparent",
          "hover:border-stream hover:glow-stream hover:bg-stream/5",
        ],
        className
      )}
      {...props}
    >
      {children}
    </motion.button>
  );
}
