"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useSpring, useTransform } from "framer-motion";

interface AnimatedNumberProps {
  value: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
  duration?: number;
  className?: string;
  colorPositive?: string;
  colorNegative?: string;
  colorNeutral?: string;
}

/**
 * Smooth animated number counter with slot-machine style transitions.
 * Automatically interpolates from old value to new value.
 */
export function AnimatedNumber({
  value,
  prefix = "",
  suffix = "",
  decimals = 2,
  duration = 0.5,
  className = "",
  colorPositive = "#00FF88",
  colorNegative = "#FF4466",
  colorNeutral = "#FFFFFF",
}: AnimatedNumberProps) {
  const [displayValue, setDisplayValue] = useState(value);
  const springValue = useSpring(value, {
    stiffness: 100,
    damping: 30,
    duration: duration * 1000,
  });

  useEffect(() => {
    springValue.set(value);
  }, [value, springValue]);

  useEffect(() => {
    const unsubscribe = springValue.on("change", (latest) => {
      setDisplayValue(latest);
    });
    return unsubscribe;
  }, [springValue]);

  // Determine color based on value
  const getColor = () => {
    if (value > 0) return colorPositive;
    if (value < 0) return colorNegative;
    return colorNeutral;
  };

  // Format number with locale and decimals
  const formatNumber = (num: number): string => {
    return num.toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  };

  return (
    <motion.span
      className={className}
      style={{ color: getColor() }}
      initial={{ opacity: 0.8 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
    >
      {prefix}
      {formatNumber(displayValue)}
      {suffix}
    </motion.span>
  );
}
