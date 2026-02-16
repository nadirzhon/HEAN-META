"use client";

import { motion } from "framer-motion";
import clsx from "clsx";

interface MetricCardProps {
  label: string;
  children: React.ReactNode;
  className?: string;
  delay?: number;
  trend?: "up" | "down" | "neutral";
}

export default function MetricCard({
  label,
  children,
  className,
  delay = 0,
  trend,
}: MetricCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3, delay, ease: "easeOut" }}
      className={clsx("glass-sm p-3 md:p-4", className)}
    >
      <p className="text-[11px] font-medium text-starlight/50 uppercase tracking-wider mb-1">
        {label}
      </p>
      <div className="flex items-baseline gap-2">
        <div className="text-lg md:text-xl font-semibold text-supernova">{children}</div>
        {trend && trend !== "neutral" && (
          <span
            className={clsx(
              "text-xs font-medium",
              trend === "up" ? "text-positive" : "text-negative"
            )}
          >
            {trend === "up" ? "+" : "-"}
          </span>
        )}
      </div>
    </motion.div>
  );
}
