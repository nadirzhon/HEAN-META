"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { ScrollControls, Scroll, useScroll, Html, PerformanceMonitor } from "@react-three/drei";
import { Suspense, useRef, useMemo, useState, useEffect } from "react";
import * as THREE from "three";
import { motion, AnimatePresence } from "framer-motion";
import GlassCard from "@/components/ui/GlassCard";
import Heading from "@/components/ui/Heading";
import clsx from "clsx";

// Tour stop data structure
interface TourStop {
  id: number;
  title: string;
  description: string;
  detail: string;
  position: [number, number, number];
  color: string;
}

const TOUR_STOPS: TourStop[] = [
  {
    id: 0,
    title: "Always Listening",
    description: "Real-time WebSocket streams from Bybit deliver ticks, order book updates, and funding rates — 24/7 with sub-second latency.",
    detail: "Supports multiple symbols simultaneously. Auto-reconnect with exponential backoff ensures zero data gaps.",
    position: [-6, 2, 0],
    color: "#00D4FF", // stream
  },
  {
    id: 1,
    title: "The Neural Highway",
    description: "A priority-queued event bus routes 38 event types between all system components with circuit-breaker protection.",
    detail: "Fast-path dispatch for SIGNAL, ORDER_REQUEST, and ORDER_FILLED bypasses normal queues for zero-latency execution.",
    position: [-3, -1, 2],
    color: "#A45BFF", // axon
  },
  {
    id: 2,
    title: "Eleven Minds, One Goal",
    description: "From ImpulseEngine's 12-layer filter cascade to FundingHarvester's rate arbitrage — each strategy competes on merit.",
    detail: "ImpulseEngine rejects 70-95% of signals through deterministic filters. Only the highest-conviction trades survive.",
    position: [0, 1.5, -1],
    color: "#00D4FF", // stream
  },
  {
    id: 3,
    title: "The Guardian",
    description: "A four-state machine — NORMAL, SOFT_BRAKE, QUARANTINE, HARD_STOP — with an automatic KillSwitch at 20% drawdown.",
    detail: "Kelly Criterion position sizing, deposit protection, and smart leverage work in concert to protect capital.",
    position: [3, -0.5, 1],
    color: "#A45BFF", // axon
  },
  {
    id: 4,
    title: "Surgical Precision",
    description: "Idempotent order routing ensures no duplicate orders. Every trade is tracked from signal to fill.",
    detail: "Retry logic with exponential backoff handles exchange rate limits. Position reconciliation runs continuously.",
    position: [6, 2, 0],
    color: "#00D4FF", // stream
  },
];

// Node component with floating animation
function TourNode({
  position,
  color,
  index,
  onToggleDetail
}: {
  position: [number, number, number];
  color: string;
  index: number;
  onToggleDetail: (index: number) => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const time = clock.getElapsedTime();
    meshRef.current.position.y = position[1] + Math.sin(time * 0.5 + index) * 0.15;
  });

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        onClick={() => onToggleDetail(index)}
        scale={hovered ? 1.2 : 1}
      >
        <sphereGeometry args={[0.3, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={hovered ? 1.2 : 0.8}
          toneMapped={false}
        />
      </mesh>

      {/* Outer glow ring */}
      <mesh scale={hovered ? 1.5 : 1.3}>
        <ringGeometry args={[0.35, 0.4, 32]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.3}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* (+) indicator */}
      <Html position={[0, 0.5, 0]} center>
        <button
          className={clsx(
            "w-11 h-11 rounded-full transition-all duration-300",
            "flex items-center justify-center text-supernova font-bold text-lg",
            "cursor-pointer",
            hovered ? "bg-axon scale-110" : "bg-space-light border border-glass-border"
          )}
          onClick={(e) => {
            e.stopPropagation();
            onToggleDetail(index);
          }}
          aria-label={`View details for ${TOUR_STOPS[index]?.title || 'tour stop'}`}
        >
          +
        </button>
      </Html>
    </group>
  );
}

// Connection tubes between nodes
function ConnectionTubes() {
  const tubesRef = useRef<THREE.Group>(null);

  useFrame(({ clock }) => {
    if (!tubesRef.current) return;
    // Subtle rotation of entire tube network
    tubesRef.current.rotation.y = Math.sin(clock.getElapsedTime() * 0.1) * 0.05;
  });

  const lines = useMemo(() => {
    return TOUR_STOPS.slice(0, -1).map((stop, i) => {
      const nextStop = TOUR_STOPS[i + 1];
      const start = new THREE.Vector3(...stop.position);
      const end = new THREE.Vector3(...nextStop.position);
      const curve = new THREE.QuadraticBezierCurve3(
        start,
        new THREE.Vector3(
          (start.x + end.x) / 2,
          (start.y + end.y) / 2 + 0.5,
          (start.z + end.z) / 2
        ),
        end
      );
      const points = curve.getPoints(50);
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({
        color: stop.color,
        transparent: true,
        opacity: 0.3,
      });
      return new THREE.Line(geometry, material);
    });
  }, []);

  return (
    <group ref={tubesRef}>
      {lines.map((lineObj, i) => (
        <primitive key={i} object={lineObj} />
      ))}
    </group>
  );
}

// Main 3D scene with camera animation
function TourScene({ activeDetail, onToggleDetail }: { activeDetail: number | null; onToggleDetail: (index: number) => void }) {
  const { camera } = useThree();
  const scroll = useScroll();

  // Camera waypoints along the tour
  const cameraPath = useMemo(() => {
    const waypoints = TOUR_STOPS.map(stop => {
      const pos = new THREE.Vector3(...stop.position);
      // Position camera in front of each node
      return new THREE.Vector3(pos.x, pos.y, pos.z + 4);
    });
    return new THREE.CatmullRomCurve3(waypoints, false, "catmullrom", 0.5);
  }, []);

  const lookAtTargets = useMemo(() => {
    return TOUR_STOPS.map(stop => new THREE.Vector3(...stop.position));
  }, []);

  useFrame(() => {
    // Interpolate camera position based on scroll progress
    const scrollProgress = scroll.offset; // 0 to 1 across all pages
    const t = scrollProgress; // Direct mapping to curve parameter

    const cameraPosition = cameraPath.getPointAt(Math.min(t, 0.999));
    camera.position.copy(cameraPosition);

    // Interpolate lookAt target
    const currentStopIndex = Math.floor(scrollProgress * (TOUR_STOPS.length - 1));
    const nextStopIndex = Math.min(currentStopIndex + 1, TOUR_STOPS.length - 1);
    const segmentProgress = (scrollProgress * (TOUR_STOPS.length - 1)) % 1;

    const currentTarget = lookAtTargets[currentStopIndex];
    const nextTarget = lookAtTargets[nextStopIndex];
    const lookAtTarget = new THREE.Vector3().lerpVectors(currentTarget, nextTarget, segmentProgress);

    camera.lookAt(lookAtTarget);
  });

  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.8} color="#A45BFF" />
      <pointLight position={[-10, -10, 5]} intensity={0.6} color="#00D4FF" />

      {/* Tour nodes */}
      {TOUR_STOPS.map((stop, i) => (
        <TourNode
          key={stop.id}
          position={stop.position}
          color={stop.color}
          index={i}
          onToggleDetail={onToggleDetail}
        />
      ))}

      {/* Connection tubes */}
      <ConnectionTubes />

      {/* Starfield background */}
      <Stars />
    </>
  );
}

// Simple starfield
function Stars() {
  const starsRef = useRef<THREE.Points>(null);

  const geometry = useMemo(() => {
    const positions = new Float32Array(500 * 3);
    for (let i = 0; i < 500; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 50;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 50;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 50;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return geo;
  }, []);

  useFrame(({ clock }) => {
    if (!starsRef.current) return;
    starsRef.current.rotation.y = clock.getElapsedTime() * 0.01;
  });

  return (
    <points ref={starsRef} geometry={geometry}>
      <pointsMaterial size={0.05} color="#C5C5DD" transparent opacity={0.6} />
    </points>
  );
}

// Text overlay for each scroll page
function TextOverlay({ stop, progress }: { stop: TourStop; progress: number }) {
  const isActive = progress > 0.3 && progress < 0.7; // Fade in/out window

  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      animate={{
        opacity: isActive ? 1 : 0,
        y: isActive ? 0 : 40
      }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="pointer-events-none"
    >
      <Heading as="h2" gradient className="mb-4">
        {stop.title}
      </Heading>
      <p className="text-starlight text-lg max-w-2xl leading-relaxed">
        {stop.description}
      </p>
    </motion.div>
  );
}

// HTML scroll content overlays
function ScrollContent({ activeDetail, onToggleDetail }: { activeDetail: number | null; onToggleDetail: (index: number) => void }) {
  const scroll = useScroll();
  const [scrollProgress, setScrollProgress] = useState(0);

  useFrame(() => {
    setScrollProgress(scroll.offset);
  });

  return (
    <Scroll html style={{ width: "100%" }}>
      {TOUR_STOPS.map((stop, i) => {
        const pageStart = i / (TOUR_STOPS.length - 1);
        const pageEnd = (i + 1) / (TOUR_STOPS.length - 1);
        const pageProgress = THREE.MathUtils.clamp(
          (scrollProgress - pageStart) / (pageEnd - pageStart),
          0,
          1
        );

        return (
          <div
            key={stop.id}
            className="h-screen flex items-center justify-start px-6 sm:px-8 md:px-24"
            style={{ position: "sticky", top: 0 }}
          >
            <TextOverlay stop={stop} progress={pageProgress} />
          </div>
        );
      })}

      {/* Detail popover */}
      <AnimatePresence>
        {activeDetail !== null && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.3 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-8 bg-space/80 backdrop-blur-sm"
            onClick={() => onToggleDetail(-1)}
          >
            <div className="pointer-events-auto" onClick={(e) => e.stopPropagation()}>
              <GlassCard className="max-w-lg">
                <div className="flex justify-between items-start mb-4">
                  <Heading as="h3" gradient>
                    {TOUR_STOPS[activeDetail].title}
                  </Heading>
                  <button
                    className="text-starlight hover:text-supernova transition-colors text-2xl leading-none"
                    onClick={() => onToggleDetail(-1)}
                  >
                    ×
                  </button>
                </div>
                <p className="text-starlight mb-4">{TOUR_STOPS[activeDetail].description}</p>
                <div className="border-t border-glass-border pt-4">
                  <p className="text-sm text-starlight/80">{TOUR_STOPS[activeDetail].detail}</p>
                </div>
              </GlassCard>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Scroll>
  );
}

// Main tour component
function TourCanvas() {
  const [activeDetail, setActiveDetail] = useState<number | null>(null);
  const [dpr, setDpr] = useState(1.5);

  const handleToggleDetail = (index: number) => {
    setActiveDetail(index === activeDetail ? null : index >= 0 ? index : null);
  };

  return (
    <Canvas
      camera={{ position: [0, 0, 5], fov: 60 }}
      gl={{ alpha: false, antialias: true }}
      dpr={dpr}
      style={{ background: "#0D0D1A" }}
    >
      <PerformanceMonitor
        onDecline={() => setDpr(1)}
        onIncline={() => setDpr(1.5)}
      >
        <Suspense fallback={null}>
          <ScrollControls pages={5} damping={0.2}>
            <TourScene activeDetail={activeDetail} onToggleDetail={handleToggleDetail} />
            <ScrollContent activeDetail={activeDetail} onToggleDetail={handleToggleDetail} />
          </ScrollControls>
        </Suspense>
      </PerformanceMonitor>
    </Canvas>
  );
}

// Main export with section wrapper
export default function InteractiveTour() {
  return (
    <section id="tour" className="relative w-full h-[500vh] hidden md:block">
      {/* Section header - positioned absolutely at top */}
      <div className="absolute top-0 left-0 right-0 z-10 pt-24 px-8 md:px-24 pointer-events-none">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <Heading as="h2" gradient className="mb-6">
            The Signal Chain
          </Heading>
          <p className="text-starlight text-base md:text-xl max-w-3xl mx-auto">
            Scroll to explore the HEAN trading system architecture — from raw market data to precision execution.
          </p>
        </motion.div>
      </div>

      {/* 3D Canvas - fixed positioning */}
      <div className="fixed top-0 left-0 w-full h-screen">
        <TourCanvas />
      </div>
    </section>
  );
}
