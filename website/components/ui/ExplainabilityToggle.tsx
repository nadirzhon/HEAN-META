"use client";

import { motion } from "framer-motion";
import { useExplainability } from "@/hooks/useExplainability";

/**
 * ExplainabilityToggle
 *
 * A glass morphism toggle switch for switching between Simple and Technical modes.
 * Fixed in the top-right corner with smooth slide animation.
 */
export default function ExplainabilityToggle() {
  const { mode, toggle } = useExplainability();

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.4 }}
      className="fixed top-4 right-4 sm:top-6 sm:right-6 z-50"
    >
      <div
        className="relative flex items-center gap-2 rounded-full px-2 py-1.5 backdrop-blur-xl"
        style={{
          background: "rgba(28, 28, 49, 0.5)",
          border: "1px solid rgba(197, 197, 221, 0.15)",
          boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3)",
        }}
      >
        {/* Simple Mode Button */}
        <button
          onClick={() => mode !== "simple" && toggle()}
          className="relative z-10 flex items-center gap-1 sm:gap-1.5 rounded-full px-2.5 py-1.5 sm:px-3 text-xs font-medium transition-colors min-h-[36px]"
          style={{
            color: mode === "simple" ? "#FFFFFF" : "#C5C5DD",
          }}
          aria-label="Simple mode"
        >
          <svg
            className="h-3.5 w-3.5"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={2}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18"
            />
          </svg>
          <span className="hidden xs:inline sm:inline">Simple</span>
        </button>

        {/* Technical Mode Button */}
        <button
          onClick={() => mode !== "technical" && toggle()}
          className="relative z-10 flex items-center gap-1 sm:gap-1.5 rounded-full px-2.5 py-1.5 sm:px-3 text-xs font-medium transition-colors min-h-[36px]"
          style={{
            color: mode === "technical" ? "#FFFFFF" : "#C5C5DD",
          }}
          aria-label="Technical mode"
        >
          <svg
            className="h-3.5 w-3.5"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={2}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5"
            />
          </svg>
          <span className="hidden xs:inline sm:inline">Technical</span>
        </button>

        {/* Sliding Background Pill */}
        <motion.div
          layoutId="mode-pill"
          className="absolute rounded-full"
          style={{
            background: "linear-gradient(135deg, #A45BFF, #00D4FF)",
            boxShadow: "0 4px 16px rgba(164, 91, 255, 0.4)",
          }}
          initial={false}
          animate={{
            left: mode === "simple" ? "4px" : "calc(50% - 2px)",
            width: mode === "simple" ? "48px" : "52px",
            height: "32px",
          }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
        />
      </div>
    </motion.div>
  );
}
