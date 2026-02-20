/**
 * Signal layer: pure functions for liquidity, flow, and composite metrics.
 * Shared schema between Analytics and Backtest:
 *   { t, liquidity_imbalance, aggression_imbalance, absorption_score }
 * Signals at time t use only data ≤ t. Backtest entry triggers on next candle open.
 */

/** Normalized signal point (shared schema). */
export function normalizeSignalPoint(snapshot, flow, depthData, liveTrades, liveConnected) {
  const t = snapshot?.ts ?? flow?.ts ?? (depthData?.ts ? new Date(depthData.ts).toISOString() : null);
  let liquidity_imbalance = null;
  let aggression_imbalance = null;
  let absorption_score = null;

  if (liveConnected && depthData?.bids?.length && depthData?.asks?.length) {
    const topN = 20;
    const bidDepth = depthData.bids[Math.min(topN - 1, depthData.bids.length - 1)]?.cum_size ?? 0;
    const askDepth = depthData.asks[Math.min(topN - 1, depthData.asks.length - 1)]?.cum_size ?? 0;
    const total = bidDepth + askDepth;
    liquidity_imbalance = total > 0 ? Math.max(-1, Math.min(1, (bidDepth - askDepth) / total)) : null;
  } else if (snapshot?.bands?.[0]) {
    liquidity_imbalance = Math.max(-1, Math.min(1, Number(snapshot.bands[0].weighted_obi) || 0));
  }

  if (liveConnected && liveTrades?.length) {
    const recent = liveTrades.slice(-100);
    let buyVol = 0, sellVol = 0;
    recent.forEach((tr) => {
      const s = Number(tr.size) || 0;
      if ((tr.side || "").toLowerCase() === "buy") buyVol += s;
      else sellVol += s;
    });
    const totalVol = buyVol + sellVol;
    aggression_imbalance = totalVol > 0 ? Math.max(-1, Math.min(1, (buyVol - sellVol) / totalVol)) : null;
    if (depthData?.bids?.length && depthData?.asks?.length && totalVol > 0) {
      const totalDepth = (depthData.bids[depthData.bids.length - 1]?.cum_size ?? 0) + (depthData.asks[depthData.asks.length - 1]?.cum_size ?? 0);
      const ratio = totalDepth / totalVol;
      absorption_score = Math.min(100, Math.max(0, (1 / (1 + 1 / (ratio || 1))) * 100));
    }
  } else if (flow) {
    aggression_imbalance = Math.max(-1, Math.min(1, Number(flow.aggressive_imbalance) || 0));
    const ratio = Number(flow.absorption_ratio) || 0;
    absorption_score = Math.min(100, Math.max(0, (1 / (1 + 1 / (ratio || 1))) * 100));
  }

  if (liquidity_imbalance === null && aggression_imbalance === null && absorption_score === null) return null;
  return { t, liquidity_imbalance, aggression_imbalance, absorption_score };
}

/**
 * @param {{ bands?: Array<{ weighted_obi?: number }> }} snapshot
 * @returns {{ liquidityValue: number } | null } -1..1, null if no top band
 */
export function deriveLiquiditySignal(snapshot) {
  const topBand = snapshot?.bands?.[0];
  if (!topBand) return null;
  const obi = Number(topBand.weighted_obi) || 0;
  return { liquidityValue: Math.max(-1, Math.min(1, obi)) };
}

/**
 * @param {{ aggressive_imbalance?: number, absorption_ratio?: number }} flow
 * @returns {{ aggressionValue: number, absorptionPercent: number } | null }
 */
export function deriveFlowSignal(flow) {
  if (!flow) return null;
  const agg = Number(flow.aggressive_imbalance) || 0;
  const ratio = Number(flow.absorption_ratio) || 0;
  const absorptionPercent = Math.min(100, Math.max(0, (1 / (1 + 1 / (ratio || 1))) * 100));
  return {
    aggressionValue: Math.max(-1, Math.min(1, agg)),
    absorptionPercent,
  };
}

/**
 * @param {{ bands?: Array<{ weighted_obi?: number }> }} snapshot
 * @param {{ absorption_ratio?: number }} flow
 * @returns {{ liquidityDefensePct: number, aggressionEfficiencyPct: number } | null }
 */
export function deriveCompositeSignal(snapshot, flow) {
  if (!snapshot?.bands?.length || !flow) return null;
  const wobi = Number(snapshot.bands[0].weighted_obi) ?? 0;
  const liquidityDefensePct = ((wobi + 1) / 2) * 100;
  const ratio = Number(flow.absorption_ratio) || 1;
  const aggressionEfficiencyPct = Math.min(100, Math.max(0, (1 / (1 + 1 / ratio)) * 100));
  return { liquidityDefensePct, aggressionEfficiencyPct };
}
