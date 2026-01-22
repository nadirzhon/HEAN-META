import { apiRequest } from "./client";

export interface KillswitchStatusResponse {
  triggered: boolean;
  reasons: string[];
  triggered_at: string | null;
  thresholds: {
    drawdown_pct: number;
    equity_drop: number;
    max_loss: number;
    risk_limit: number;
  };
  current_metrics: {
    current_drawdown_pct: number;
    current_equity: number;
    max_drawdown_pct: number;
    peak_equity?: number;
  };
}

export async function fetchKillswitchStatus(): Promise<KillswitchStatusResponse> {
  return apiRequest<KillswitchStatusResponse>("/risk/killswitch/status");
}

export async function resetKillswitch(confirm: boolean = false): Promise<{ status: string; message: string }> {
  return apiRequest<{ status: string; message: string }>("/risk/killswitch/reset", {
    method: "POST",
    body: JSON.stringify({ confirm }),
  });
}