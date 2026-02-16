"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import GlassCard from "@/components/ui/GlassCard";
import Heading from "@/components/ui/Heading";
import Paragraph from "@/components/ui/Paragraph";
import { useExplainability } from "@/hooks/useExplainability";
import clsx from "clsx";

interface FeatureTab {
  id: string;
  label: string;
  iconColor: string;
  icon: React.ReactNode;
  title: string;
  description: [string, string]; // [simple, technical]
  bullets: [string, string][]; // [[simple, technical], ...]
}

const BrainIcon = (
  <svg
    className="w-12 h-12"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
    aria-hidden="true"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18"
    />
    <circle cx="12" cy="6" r="1.5" fill="currentColor" />
    <circle cx="8" cy="9" r="1" fill="currentColor" />
    <circle cx="16" cy="9" r="1" fill="currentColor" />
  </svg>
);

const ShieldIcon = (
  <svg
    className="w-12 h-12"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
    aria-hidden="true"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z"
    />
  </svg>
);

const DNAIcon = (
  <svg
    className="w-12 h-12"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
    aria-hidden="true"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c.332.332.487.811.422 1.29l-.338 2.508a1.5 1.5 0 01-1.492 1.3h-1.06M5 14.5l-1.402 1.402A1.5 1.5 0 003 17.192l.338 2.508A1.5 1.5 0 004.83 21h1.06"
    />
    <circle cx="7" cy="6" r="1" fill="currentColor" />
    <circle cx="17" cy="6" r="1" fill="currentColor" />
    <circle cx="7" cy="18" r="1" fill="currentColor" />
    <circle cx="17" cy="18" r="1" fill="currentColor" />
  </svg>
);

const EyeIcon = (
  <svg
    className="w-12 h-12"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
    aria-hidden="true"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z"
    />
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
    />
  </svg>
);

const FEATURES: FeatureTab[] = [
  {
    id: "adaptive",
    label: "Adaptive Intelligence",
    iconColor: "text-axon",
    icon: BrainIcon,
    title: "Adaptive Intelligence",
    description: [
      "HEAN reads the market like a doctor reads vital signs \u2014 temperature, energy, and flow \u2014 to understand what's happening right now.",
      "MarketGenome synthesizes 6 data dimensions \u2014 regime, volatility, thermodynamics, liquidity, momentum, and funding \u2014 into a unified market consciousness updated every 10 seconds.",
    ],
    bullets: [
      [
        "Checks the market's mood across 6 different angles",
        "Real-time regime detection across 6 data dimensions",
      ],
      [
        "Measures market energy and chaos levels",
        "Market temperature and entropy thermodynamics",
      ],
      [
        "Identifies if money is flowing in, peaking, or flowing out",
        "Phase detection: accumulation, markup, distribution, markdown",
      ],
    ],
  },
  {
    id: "antifragile",
    label: "Anti-Fragile Core",
    iconColor: "text-stream",
    icon: ShieldIcon,
    title: "Anti-Fragile Core",
    description: [
      "Every hour, HEAN tests itself against 7 worst-case scenarios. If it can't survive a crash, it automatically reduces risk.",
      "DoomsdaySandbox simulates 7 catastrophic scenarios against your live portfolio every hour. Survival scoring from 0.0 to 1.0 with automatic protective measures.",
    ],
    bullets: [
      [
        "Simulates 7 disaster scenarios every hour",
        "7 catastrophic scenarios simulated hourly against live portfolio",
      ],
      [
        "Gives itself a survival score and acts on it",
        "Survival scoring from 0.0 to 1.0 with threshold-based responses",
      ],
      [
        "Automatically steps on the brakes when danger is high",
        "Automatic risk reduction on low survival scores via RiskGovernor state machine",
      ],
    ],
  },
  {
    id: "evolving",
    label: "Evolving Portfolio",
    iconColor: "text-axon",
    icon: DNAIcon,
    title: "Evolving Portfolio",
    description: [
      "Strategies that stop working get put to sleep. New ones are born from the best performers through digital evolution.",
      "MetaStrategyBrain manages strategy lifecycles through Active, Reduced, Hibernated, and Terminated states \u2014 detecting alpha decay and triggering genetic evolution.",
    ],
    bullets: [
      [
        "Winning strategies breed; losing ones hibernate",
        "Genetic algorithm strategy evolution via Symbiont X genome lab",
      ],
      [
        "Spots when a strategy starts losing its edge",
        "Alpha decay detection with automatic state transitions",
      ],
      [
        "Strategies go through life stages: active, reduced, sleeping, retired",
        "Strategy lifecycle: Active \u2192 Reduced \u2192 Hibernated \u2192 Terminated",
      ],
    ],
  },
  {
    id: "transparency",
    label: "Full Transparency",
    iconColor: "text-stream",
    icon: EyeIcon,
    title: "Full Transparency",
    description: [
      "Every decision HEAN makes is visible and explainable. No black boxes — you always know why a trade was taken or rejected.",
      "Complete observability with per-strategy telemetry, signal attribution, and decision explainability. Every trade includes confidence scores, filter cascade results, and the exact conditions that triggered it.",
    ],
    bullets: [
      [
        "See exactly why every trade was made or rejected",
        "Full signal attribution with 12-layer filter cascade rejection reasons",
      ],
      [
        "Real-time dashboard shows everything happening inside",
        "Per-strategy confidence metrics, win rates, and alpha decay indicators",
      ],
      [
        "Switch between simple and technical views anytime",
        "Dual-mode explainability system with localStorage-persisted preferences",
      ],
    ],
  },
];

export default function Features() {
  const [activeTab, setActiveTab] = useState(0);
  const { text, mode } = useExplainability();
  const feature = FEATURES[activeTab];

  return (
    <section className="relative py-32 aurora-bg">
      <div className="mx-auto max-w-5xl px-6 lg:px-8">
        {/* Section Header */}
        <div className="mx-auto max-w-2xl text-center mb-16">
          <Heading as="h2" gradient className="mb-8">
            {text("How HEAN Thinks", "Digital Organism Architecture")}
          </Heading>
          <Paragraph size="lg" className="text-starlight/80">
            {text(
              "Four smart systems that work together \u2014 learning from markets and getting better over time.",
              "Four interconnected systems that learn, adapt, and evolve \u2014 transforming market chaos into structured edge."
            )}
          </Paragraph>
        </div>

        {/* Tab Buttons */}
        <div className="flex flex-wrap items-center justify-center gap-3 sm:gap-4 mb-12">
          {FEATURES.map((f, i) => (
            <button
              key={f.id}
              onClick={() => setActiveTab(i)}
              className={clsx(
                "relative rounded-full px-4 py-2.5 sm:px-6 sm:py-3 text-xs sm:text-sm font-medium transition-colors duration-300 cursor-pointer min-h-[44px]",
                activeTab === i
                  ? "text-supernova"
                  : "text-starlight/60 hover:text-starlight"
              )}
            >
              {activeTab === i && (
                <motion.div
                  layoutId="feature-tab-indicator"
                  className="absolute inset-0 rounded-full"
                  style={{
                    background: "rgba(164, 91, 255, 0.15)",
                    border: "1px solid rgba(164, 91, 255, 0.3)",
                  }}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
              <span className="relative z-10">{f.label}</span>
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3 }}
          >
            <GlassCard>
              <div className="flex flex-col md:flex-row gap-6 md:gap-8 items-start">
                {/* Icon + Title */}
                <div className="flex flex-col items-center md:items-start gap-3 md:gap-4 w-full md:w-auto md:min-w-[200px]">
                  <div className={feature.iconColor}>{feature.icon}</div>
                  <Heading as="h3">{feature.title}</Heading>
                </div>

                {/* Description + Bullets */}
                <div className="flex-1">
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={`desc-${activeTab}-${mode}`}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <Paragraph className="text-starlight/80 mb-8">
                        {text(feature.description[0], feature.description[1])}
                      </Paragraph>
                    </motion.div>
                  </AnimatePresence>

                  <ul className="space-y-3 sm:space-y-4">
                    {feature.bullets.map((bullet, i) => (
                      <li key={i} className="flex items-start gap-2 sm:gap-3">
                        <span className="mt-1.5 h-2 w-2 rounded-full bg-stream shrink-0" />
                        <AnimatePresence mode="wait">
                          <motion.span
                            key={`bullet-${activeTab}-${i}-${mode}`}
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.2 }}
                            className="text-sm text-starlight/70"
                          >
                            {text(bullet[0], bullet[1])}
                          </motion.span>
                        </AnimatePresence>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </GlassCard>
          </motion.div>
        </AnimatePresence>
      </div>
    </section>
  );
}
