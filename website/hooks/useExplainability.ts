"use client";

import { useExplainabilityContext } from "@/lib/explainability/ExplainabilityContext";
import type { ExplainabilityMode } from "@/lib/explainability/ExplainabilityContext";

export function useExplainability() {
  const { mode, toggle } = useExplainabilityContext();

  const text = (simple: string, technical: string): string => {
    return mode === "simple" ? simple : technical;
  };

  return {
    mode,
    toggle,
    text,
  };
}
