"use client";

import { useRef } from "react";
import { motion, AnimatePresence, useInView } from "framer-motion";
import Heading from "@/components/ui/Heading";
import Paragraph from "@/components/ui/Paragraph";
import Button from "@/components/ui/Button";
import { useExplainability } from "@/hooks/useExplainability";

export default function CTA() {
  const sectionRef = useRef<HTMLElement>(null);
  const isInView = useInView(sectionRef, { once: true, margin: "-60px" });
  const { text, mode } = useExplainability();

  return (
    <section
      ref={sectionRef}
      id="cta"
      className="relative overflow-hidden py-36 md:py-44"
    >
      {/* Radial axon glow from center */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background: [
            "radial-gradient(ellipse at 50% 50%, rgba(164,91,255,0.12) 0%, transparent 55%)",
            "radial-gradient(ellipse at 50% 70%, rgba(0,212,255,0.06) 0%, transparent 50%)",
          ].join(", "),
        }}
      />

      {/* Faint top border line */}
      <div
        className="pointer-events-none absolute top-0 left-1/2 h-px w-[60%] -translate-x-1/2"
        style={{
          background:
            "linear-gradient(90deg, transparent, rgba(164,91,255,0.3), rgba(0,212,255,0.3), transparent)",
        }}
      />

      <div className="relative mx-auto max-w-2xl px-6 text-center">
        {/* Heading */}
        <motion.div
          initial={{ opacity: 0, y: 28 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 28 }}
          transition={{ duration: 0.65, ease: "easeOut" }}
        >
          <Heading as="h2" gradient>
            {text("Join the Future of Smart Investing", "Become Part of the Future")}
          </Heading>
        </motion.div>

        {/* Description */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.6, delay: 0.15, ease: "easeOut" }}
          className="mt-6"
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={`cta-desc-${mode}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <Paragraph size="lg" className="text-starlight/80">
                {text(
                  "HEAN is building the future of smart investing. Join the waitlist to see what autonomous trading can do.",
                  "HEAN is redefining what autonomous trading systems can achieve. Request early access to experience the next evolution in financial intelligence."
                )}
              </Paragraph>
            </motion.div>
          </AnimatePresence>
        </motion.div>

        {/* Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 16 }}
          transition={{ duration: 0.55, delay: 0.3, ease: "easeOut" }}
          className="mt-8 sm:mt-10 flex flex-col items-stretch sm:items-center justify-center gap-3 sm:gap-4 sm:flex-row w-full sm:w-auto max-w-md sm:max-w-none mx-auto"
        >
          <Button variant="primary" className="w-full sm:w-auto">Request Access</Button>
          <Button variant="outline" className="w-full sm:w-auto">View Documentation</Button>
        </motion.div>
      </div>
    </section>
  );
}
