"use client";

import { motion } from "framer-motion";
import Heading from "@/components/ui/Heading";
import Paragraph from "@/components/ui/Paragraph";
import { useExplainability } from "@/hooks/useExplainability";
import clsx from "clsx";

interface Advantage {
  icon: React.ReactNode;
  iconColor: string;
  title: string;
  description: [string, string]; // [simple, technical]
}

const AutomationIcon = (
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
      d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z"
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

const ADVANTAGES: Advantage[] = [
  {
    icon: AutomationIcon,
    iconColor: "text-axon",
    title: "AI-Powered Autonomy",
    description: [
      "HEAN monitors markets 24/7 and executes trades autonomously — no human intervention needed.",
      "Fully autonomous event-driven architecture with real-time market monitoring and order execution — zero human intervention required.",
    ],
  },
  {
    icon: DNAIcon,
    iconColor: "text-stream",
    title: "Adaptive Strategies",
    description: [
      "11 concurrent strategies evolve through genetic algorithms, automatically adapting to any market regime.",
      "11 concurrent strategies with genetic algorithm evolution via Symbiont X genome lab — automatic adaptation to regime shifts through crossover and mutation.",
    ],
  },
  {
    icon: ShieldIcon,
    iconColor: "text-axon",
    title: "Bulletproof Risk Control",
    description: [
      "A four-state risk machine with automatic KillSwitch protects capital even in worst-case scenarios.",
      "Four-state RiskGovernor state machine (NORMAL → SOFT_BRAKE → QUARANTINE → HARD_STOP) with automatic KillSwitch triggering on 20% drawdown threshold.",
    ],
  },
];

export default function Summary() {
  const { text } = useExplainability();

  return (
    <section className="relative py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <h2 className="sr-only">Key Advantages</h2>
        <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
          {ADVANTAGES.map((advantage, index) => (
            <motion.div
              key={advantage.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{
                duration: 0.6,
                delay: index * 0.15,
              }}
              className="glass rounded-2xl p-8"
            >
              <div className="mb-6">
                <div className={clsx(advantage.iconColor, "mb-4")}>
                  {advantage.icon}
                </div>
                <Heading as="h3" className="mb-4">
                  {advantage.title}
                </Heading>
              </div>
              <Paragraph size="sm" className="text-starlight/80">
                {text(advantage.description[0], advantage.description[1])}
              </Paragraph>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
