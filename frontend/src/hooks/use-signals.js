import { useMemo } from "react";
import { normalizeSignalPoint } from "@/signals";

/**
 * Normalized signals for Analytics ↔ Backtest continuity.
 * Returns shared schema: { t, liquidity_imbalance, aggression_imbalance, absorption_score }.
 * Signals at time t use only data ≤ t; backtest entry triggers on next candle open.
 *
 * @param {{ bands?: Array<{ weighted_obi?: number }>, ts?: string } | null} snapshot
 * @param {{ aggressive_imbalance?: number, absorption_ratio?: number, ts?: string } | null} flow
 * @param {{ bids?: Array<{ cum_size?: number }>, asks?: Array<{ cum_size?: number }>, ts?: string } | null} depthData
 * @param {Array<{ side?: string, size?: number }>} liveTrades
 * @param {boolean} liveConnected
 */
export function useSignals(snapshot, flow, depthData, liveTrades, liveConnected) {
  return useMemo(() => {
    const signal = normalizeSignalPoint(snapshot, flow, depthData, liveTrades ?? [], !!liveConnected);
    return { signal };
  }, [snapshot, flow, depthData, liveTrades, liveConnected]);
}
