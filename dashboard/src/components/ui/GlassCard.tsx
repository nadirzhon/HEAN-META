"use client";

import { motion } from "framer-motion";
import clsx from "clsx";

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  delay?: number;
  padding?: "sm" | "md" | "lg";
  hover?: boolean;
}

export default function GlassCard({
  children,
  className,
  delay = 0,
  padding = "md",
  hover = false,
}: GlassCardProps) {
  const paddings = { sm: "p-3", md: "p-4 md:p-5", lg: "p-5 md:p-6" };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay, ease: "easeOut" }}
      className={clsx(
        "glass",
        paddings[padding],
        hover && "hover:border-axon/30 hover:shadow-[0_0_20px_rgba(164,91,255,0.1)] transition-all duration-300",
        className
      )}
    >
      {children}
    </motion.div>
  );
}
