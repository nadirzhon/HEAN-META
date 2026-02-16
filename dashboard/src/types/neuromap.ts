/**
 * Type definitions for the Neuro-Map visualization system.
 * Models the HEAN architecture as a directed graph of processing nodes
 * connected by an EventBus.
 */

export type NodeColor = "axon" | "stream";

export type NodeStatus = "active" | "degraded" | "error" | "idle";

export interface SystemNode {
  id: string;
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  color: NodeColor;
  status: NodeStatus;
  eventsPerSec: number;
  description?: string;
}

export interface NodeConnection {
  id: string;
  sourceId: string;
  targetId: string;
  /** Whether impulses are currently flowing on this connection */
  active: boolean;
}

export interface Impulse {
  id: string;
  connectionId: string;
  /** 0..1 progress along the path */
  progress: number;
  /** Timestamp when the impulse was created */
  createdAt: number;
}

export type AgentRole = "brain" | "council" | "risk";

export interface AgentThought {
  id: string;
  agent: AgentRole;
  message: string;
  timestamp: number;
  /** Whether this thought is still being "typed" */
  isStreaming: boolean;
}

export interface NeuroMapState {
  nodes: SystemNode[];
  connections: NodeConnection[];
  impulses: Impulse[];
  agentThoughts: AgentThought[];
  totalEventsPerSec: number;
}

/**
 * Color palette constants for the Neural Midnight theme.
 */
export const NEURO_COLORS = {
  space: "#0D0D1A",
  axon: "#A45BFF",
  stream: "#00D4FF",
  starlight: "#C5C5DD",
  supernova: "#FFFFFF",
  glassBg: "rgba(28,28,49,0.5)",
  glassBorder: "rgba(255,255,255,0.1)",
  statusActive: "#00FF88",
  statusDegraded: "#FFD600",
  statusError: "#FF3355",
  statusIdle: "#555577",
} as const;

/**
 * Resolves a NodeColor token to its hex value.
 */
export function resolveColor(color: NodeColor): string {
  return color === "axon" ? NEURO_COLORS.axon : NEURO_COLORS.stream;
}
