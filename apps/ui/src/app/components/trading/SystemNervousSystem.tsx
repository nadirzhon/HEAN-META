import { useEffect, useState } from "react";
import { motion } from "motion/react";

interface SystemNervousSystemProps {
  eventBusLoad: number; // 0-100
  tickRate: number; // ticks per second
  decisionRate: number; // decisions per minute
}

export function SystemNervousSystem({ eventBusLoad, tickRate, decisionRate }: SystemNervousSystemProps) {
  const [pulses, setPulses] = useState<Array<{ id: number; x: number; y: number; delay: number }>>([]);

  useEffect(() => {
    // Generate pulse positions based on system load
    const pulseCount = Math.floor((eventBusLoad / 100) * 12) + 3;
    const newPulses = Array.from({ length: pulseCount }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      delay: Math.random() * 2,
    }));
    setPulses(newPulses);
  }, [eventBusLoad]);

  const getStressLevel = () => {
    if (eventBusLoad < 30) return "calm";
    if (eventBusLoad < 70) return "active";
    return "stressed";
  };

  const stressLevel = getStressLevel();
  const pulseColor = 
    stressLevel === "calm" ? "var(--trading-green)" :
    stressLevel === "active" ? "var(--trading-cyan)" :
    "var(--trading-amber)";

  return (
    <div className="absolute inset-0 pointer-events-none opacity-20">
      <svg width="100%" height="100%" className="absolute inset-0">
        <defs>
          <filter id="nervousGlow">
            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
          
          <radialGradient id="pulseGradient">
            <stop offset="0%" stopColor={pulseColor} stopOpacity="0.8" />
            <stop offset="100%" stopColor={pulseColor} stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* Connection lines */}
        {pulses.map((pulse, idx) => {
          const nextPulse = pulses[(idx + 1) % pulses.length];
          return (
            <line
              key={`line-${pulse.id}`}
              x1={`${pulse.x}%`}
              y1={`${pulse.y}%`}
              x2={`${nextPulse.x}%`}
              y2={`${nextPulse.y}%`}
              stroke={pulseColor}
              strokeWidth="1"
              opacity="0.1"
              strokeDasharray="4 4"
            />
          );
        })}

        {/* Pulse nodes */}
        {pulses.map((pulse) => (
          <g key={pulse.id}>
            <motion.circle
              cx={`${pulse.x}%`}
              cy={`${pulse.y}%`}
              r="4"
              fill={pulseColor}
              filter="url(#nervousGlow)"
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ 
                scale: [0.5, 1.2, 0.5],
                opacity: [0.3, 0.8, 0.3]
              }}
              transition={{
                duration: stressLevel === "calm" ? 3 : stressLevel === "active" ? 2 : 1.5,
                repeat: Infinity,
                delay: pulse.delay,
              }}
            />
            <motion.circle
              cx={`${pulse.x}%`}
              cy={`${pulse.y}%`}
              r="8"
              fill="none"
              stroke={pulseColor}
              strokeWidth="1"
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ 
                scale: [0.5, 1.5, 0.5],
                opacity: [0.5, 0, 0.5]
              }}
              transition={{
                duration: stressLevel === "calm" ? 3 : stressLevel === "active" ? 2 : 1.5,
                repeat: Infinity,
                delay: pulse.delay + 0.5,
              }}
            />
          </g>
        ))}
      </svg>

      {/* Breathing effect overlay */}
      <motion.div
        className="absolute inset-0"
        style={{
          background: `radial-gradient(circle at 50% 50%, ${pulseColor} 0%, transparent 70%)`,
        }}
        animate={{
          opacity: stressLevel === "calm" ? [0.02, 0.05, 0.02] : stressLevel === "active" ? [0.05, 0.1, 0.05] : [0.1, 0.15, 0.1],
        }}
        transition={{
          duration: stressLevel === "calm" ? 4 : stressLevel === "active" ? 2.5 : 1.5,
          repeat: Infinity,
        }}
      />

      {/* System metrics overlay */}
      <div className="absolute bottom-4 right-4 font-mono text-xs">
        <div style={{ color: pulseColor }} className="opacity-60">
          {tickRate} t/s • {decisionRate} d/m
        </div>
      </div>
    </div>
  );
}
