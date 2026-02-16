"use client";

import clsx from "clsx";

type BadgeVariant = "normal" | "warning" | "danger" | "info" | "success";

const RISK_STATE_MAP: Record<string, BadgeVariant> = {
  NORMAL: "success",
  SOFT_BRAKE: "warning",
  QUARANTINE: "danger",
  HARD_STOP: "danger",
};

interface StatusBadgeProps {
  label: string;
  variant?: BadgeVariant;
  /** Auto-map HEAN risk states to colors */
  riskState?: string;
  size?: "sm" | "md";
  pulse?: boolean;
}

export default function StatusBadge({
  label,
  variant,
  riskState,
  size = "sm",
  pulse = false,
}: StatusBadgeProps) {
  const resolvedVariant = riskState ? (RISK_STATE_MAP[riskState] || "info") : (variant || "info");

  const colors: Record<BadgeVariant, string> = {
    normal: "bg-starlight/10 text-starlight border-starlight/20",
    success: "bg-positive/10 text-positive border-positive/30",
    warning: "bg-warning/10 text-warning border-warning/30",
    danger: "bg-negative/10 text-negative border-negative/30",
    info: "bg-stream/10 text-stream border-stream/30",
  };

  const sizes = {
    sm: "text-[10px] px-2 py-0.5",
    md: "text-xs px-2.5 py-1",
  };

  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full border font-medium uppercase tracking-wide",
        colors[resolvedVariant],
        sizes[size],
        pulse && "animate-pulse"
      )}
    >
      {pulse && (
        <span
          className={clsx(
            "w-1.5 h-1.5 rounded-full",
            resolvedVariant === "success" && "bg-positive",
            resolvedVariant === "warning" && "bg-warning",
            resolvedVariant === "danger" && "bg-negative",
            resolvedVariant === "info" && "bg-stream",
            resolvedVariant === "normal" && "bg-starlight"
          )}
        />
      )}
      {label}
    </span>
  );
}
