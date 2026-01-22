import { EventEnvelope } from "@/app/api/client";

export type EventCategory = "system" | "orders" | "positions" | "risk" | "strategies" | "other";

export interface OrderRow {
  id: string;
  symbol: string;
  side: "BUY" | "SELL";
  price?: number;
  size?: number;
  filled?: number;
  status?: string;
  strategyId?: string;
  createdAt?: Date;
  type?: string;
  takeProfit?: number;
  stopLoss?: number;
}

export interface EventFeedItem extends EventEnvelope {
  id: string;
  category: EventCategory;
  message: string;
}

export interface ChartMarker {
  time: number;
  position: "aboveBar" | "belowBar";
  color: string;
  shape: "arrowUp" | "arrowDown" | "circle";
  text: string;
}
