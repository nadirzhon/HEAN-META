"use client";

import { motion, AnimatePresence } from "framer-motion";
import dynamic from "next/dynamic";
import Button from "@/components/ui/Button";
import { useExplainability } from "@/hooks/useExplainability";

// Lazy-load Three.js component with SSR disabled + animated placeholder
const NeuralNetwork = dynamic(() => import("@/lib/three/NeuralNetwork"), {
  ssr: false,
  loading: () => (
    <div className="absolute inset-0 bg-space">
      <div className="absolute inset-0 aurora-bg animate-pulse opacity-40" />
    </div>
  ),
});

export default function Hero() {
  const { text, mode } = useExplainability();

  return (
    <section className="relative min-h-screen w-full overflow-hidden bg-space">
      {/* 3D Neural Network Background */}
      <div className="absolute inset-0 z-0">
        <NeuralNetwork />
        {/* Dark gradient overlay for text readability */}
        <div className="absolute inset-0 bg-gradient-to-b from-space/60 via-space/80 to-space" />
      </div>

      {/* Hero Content */}
      <div className="relative z-10 flex min-h-screen flex-col items-center justify-center px-6 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: "easeOut" }}
          className="max-w-5xl space-y-8"
        >
          {/* Main Headline */}
          <div className="space-y-6 sm:space-y-8">
            <h1 className="text-6xl font-bold tracking-tight text-supernova sm:text-7xl md:text-8xl lg:text-9xl">
              HEAN
            </h1>
            <AnimatePresence mode="wait">
              <motion.p
                key={`subtitle-${mode}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
                className="gradient-text text-xl font-semibold sm:text-2xl md:text-3xl lg:text-4xl"
              >
                {text(
                  "Trading intelligence that thinks for itself — in any market.",
                  "Autonomous trading intelligence that adapts, evolves, and thrives — in any market condition."
                )}
              </motion.p>
            </AnimatePresence>
          </div>

          {/* Body Text */}
          <AnimatePresence mode="wait">
            <motion.p
              key={`body-${mode}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="mx-auto max-w-2xl text-base leading-relaxed text-starlight sm:text-lg md:text-xl"
            >
              {text(
                "HEAN watches the markets 24/7, makes smart decisions, and protects your money — all without human emotions getting in the way.",
                "Event-driven architecture with real-time regime detection, genetic strategy evolution, and anti-fragile risk management across 11 concurrent strategies."
              )}
            </motion.p>
          </AnimatePresence>

          {/* CTA Button */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.6, duration: 0.8, ease: "easeOut" }}
            className="pt-6 sm:pt-8"
          >
            <a href="#features">
              <Button variant="primary" className="text-sm sm:text-base">
                Explore the System
              </Button>
            </a>
          </motion.div>
        </motion.div>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2, duration: 1 }}
          className="absolute bottom-8 sm:bottom-16 left-1/2 -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            className="flex flex-col items-center gap-4 sm:gap-8 text-starlight/60"
          >
            <span className="text-xs sm:text-sm uppercase tracking-widest">Scroll</span>
            <svg
              className="h-6 w-6"
              fill="none"
              strokeWidth="2"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="m19 9-7 7-7-7"
              />
            </svg>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
