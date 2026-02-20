import React, { useEffect, useMemo, useRef,useState } from "react";
import { api } from "@/api/client";
import { useLive } from "@/hooks/use-live";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Checkbox } from "@/components/ui/checkbox";
import { PriceChart } from "@/components/PriceChart";
import { ChartHelp } from "@/components/ChartHelp";
import { SpectrumGauge, PercentGauge, VelocityArrows } from "@/components/RegimeGauge";
import { useSignals } from "@/hooks/use-signals";
import { Clock } from "lucide-react";
import {
  convertDateTimeLocalValue,
  dateTimeLocalValueToDate,
  dateToDateTimeLocalValue,
  formatTime,
  getStoredTimezone,
  setStoredTimezone,
} from "@/utils/time";

// Hard caps to prevent runaway memory (reject older data; keep newest only).
const PRICE_CANDLES_MAX = 20000;
const FLOW_SERIES_POINTS_MAX = 50000;
const FUNDING_POINTS_MAX = 2000;
const OI_POINTS_MAX = 2000;

// Single source of truth: price timeframe → REST bin_size and live candle bucket (seconds).
const PRICE_TIMEFRAME_INTERVAL = {
  "1m": 60,
  "5m": 300,
  "15m": 900,
  "1h": 3600,
  "4h": 14400,
  "1d": 86400,
  "1w": 604800,
};

// Max bars we keep for "Max" range (must not exceed PRICE_CANDLES_MAX).
const PRICE_CANDLES_SOFT_CAP = 20000;

// Range options for price history (label, days back from now). "Max" uses timeframe-dependent cap for max history.
const PRICE_RANGE_OPTIONS = [
  { value: "7d", label: "7 days", days: 7 },
  { value: "30d", label: "30 days", days: 30 },
  { value: "90d", label: "90 days", days: 90 },
  { value: "1y", label: "1 year", days: 365 },
  { value: "max", label: "Max (all available)", days: null },
];

// Memoized tooltip style to avoid inline object recreation (reduces Recharts re-renders / memory pressure).
const TOOLTIP_CONTENT_STYLE = { background: "hsl(var(--popover))", borderRadius: 12, border: "1px solid hsl(var(--border))" };

function fmt(n, digits = 2) {
  if (n === null || n === undefined) return "—";
  if (Number.isNaN(Number(n))) return "—";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function colorForHeat(v) {
  // v: 0..1
  const x = clamp(v, 0, 1);
  // Map to HSL between teal->violet
  const hue = 175 + (285 - 175) * x;
  const sat = 80;
  const light = 20 + 22 * x;
  return `hsl(${hue} ${sat}% ${light}%)`;
}

export default function Analytics() {
  const isGuest = !!localStorage.getItem("guest");
  const live = useLive();
  const didConnectLiveRef = useRef(false);
  const didInitRef = useRef(false);
  const reconnectTimerRef = useRef(null);
  const [symbols, setSymbols] = useState([]);
  const [symbol, setSymbol] = useState("XBTUSD");
  const initialTz = getStoredTimezone("utc");

  // shared controls
  const [depth, setDepth] = useState(50);
  const [bands, setBands] = useState("10,25,100");

  // timeframes
  const [flowMinutes, setFlowMinutes] = useState(60);
  const [flowMaxDays, setFlowMaxDays] = useState(365);
  const [preset, setPreset] = useState("60m");

  const [priceTimeframe, setPriceTimeframe] = useState("1m");
  const [priceRange, setPriceRange] = useState("30d");
  const [showVolumeOverlay, setShowVolumeOverlay] = useState(false);
  const [showCvdOverlay, setShowCvdOverlay] = useState(false);
  const [useUTC, setUseUTC] = useState(initialTz === "utc");

  const [customStart, setCustomStart] = useState(() => {
    const d = new Date(Date.now() - 1000 * 60 * 60);
    return dateToDateTimeLocalValue(d, initialTz === "utc");
  });
  const [customEnd, setCustomEnd] = useState(() => {
    const d = new Date();
    return dateToDateTimeLocalValue(d, initialTz === "utc");
  });

  const [snapshot, setSnapshot] = useState(null);
  const [flow, setFlow] = useState(null);
  const [priceCandles, setPriceCandles] = useState([]);

  const [depthData, setDepthData] = useState(null);
  const [flowSeries, setFlowSeries] = useState(null);

  const [funding, setFunding] = useState(null);
  const [openInterest, setOpenInterest] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [priceChartLoading, setPriceChartLoading] = useState(false);

  // Normalized signals (shared schema with Backtest): { t, liquidity_imbalance, aggression_imbalance, absorption_score }.
  // Signals at time t use only data ≤ t; backtest entry triggers on next candle open.
  const { signal: normalizedSignal } = useSignals(snapshot, flow, depthData, live.trades, live.connected);

  const liveOrderbook = live.orderbook;

  const livePrice = useMemo(() => {
    if (liveOrderbook?.mid == null || liveOrderbook?.ts == null) return null;
    return { price: Number(liveOrderbook.mid), timestamp: liveOrderbook.ts };
  }, [liveOrderbook?.mid, liveOrderbook?.ts]);

  // Only feed live ticks to the price chart when the WS symbol matches the selected symbol.
  // Otherwise we'd mix e.g. XBT prices with XRP candles and break the scale.
  const livePriceForChart = useMemo(() => {
    const wsSymbol = liveOrderbook?.symbol ?? live.status?.symbol;
    if (!wsSymbol || wsSymbol !== symbol) return null;
    return livePrice;
  }, [symbol, liveOrderbook?.symbol, live.status?.symbol, livePrice]);

  const lastLiveApplyRef = React.useRef(0);
  const [selectedTab, setSelectedTab] = useState("price");

  // Persist timezone preference across reloads.
  useEffect(() => {
    setStoredTimezone(useUTC ? "utc" : "local");
  }, [useUTC]);

  function onTimezoneChange(next) {
    const nextUseUTC = next === "utc";
    // Convert existing datetime-local inputs to preserve the same instant.
    setCustomStart((v) => convertDateTimeLocalValue(v, useUTC, nextUseUTC));
    setCustomEnd((v) => convertDateTimeLocalValue(v, useUTC, nextUseUTC));
    setUseUTC(nextUseUTC);
  }

  // Live data: only push to React state when price or orderbook tab is visible to avoid memory churn.
  useEffect(() => {
    if (!liveOrderbook?.bids?.length || !liveOrderbook?.asks?.length) return;
    if (selectedTab !== "price" && selectedTab !== "orderbook") return;

    const now = Date.now();
    if (now - lastLiveApplyRef.current < 250) return; // ~4 updates/sec max
    lastLiveApplyRef.current = now;

    // Market top-of-book
    setSnapshot((prev) => {
      const bandsPrev = prev?.bands || [];
      const next = {
        symbol: liveOrderbook.symbol,
        ts: liveOrderbook.ts,
        best_bid: liveOrderbook.best_bid,
        best_ask: liveOrderbook.best_ask,
        mid: liveOrderbook.mid,
        spread: liveOrderbook.spread,
        bands: bandsPrev,
      };

      // If unchanged, avoid redundant updates
      if (
        prev &&
        prev.symbol === next.symbol &&
        prev.ts === next.ts &&
        prev.best_bid === next.best_bid &&
        prev.best_ask === next.best_ask &&
        prev.mid === next.mid &&
        prev.spread === next.spread
      ) {
        return prev;
      }
      return next;
    });

    // Depth chart source (convert to cumulative)
    const bids = liveOrderbook.bids;
    const asks = liveOrderbook.asks;

    let cumB = 0;
    const bidCum = bids.map((p) => {
      cumB += Number(p.size || 0);
      return { price: Number(p.price), cum_size: cumB };
    });

    let cumA = 0;
    const askCum = asks.map((p) => {
      cumA += Number(p.size || 0);
      return { price: Number(p.price), cum_size: cumA };
    });

    setDepthData((prev) => {
      const next = { symbol: liveOrderbook.symbol, ts: liveOrderbook.ts, bids: bidCum, asks: askCum };
      if (prev?.ts === next.ts && prev?.symbol === next.symbol) return prev;
      return next;
    });
  }, [liveOrderbook, selectedTab]);

  const [pollingEnabled, setPollingEnabled] = useState(false);
  const [pollIntervalSec, setPollIntervalSec] = useState(5);

  const priceSeries = useMemo(() => {
    return priceCandles.map((c) => ({ t: c.timestamp, close: c.close }));
  }, [priceCandles]);

  const [showPollingMenu, setShowPollingMenu] = useState(false);

  const latestCandle = useMemo(() => {
    if (!priceCandles.length) return null;
    return priceCandles[priceCandles.length - 1];
  }, [priceCandles]);

  const regimeMetrics = useMemo(() => {
    const useLiveData = live.connected;
    if (!useLiveData && !snapshot) return null;

    let liquidityValue = 0;
    let aggressionValue = 0;
    let absorptionPercent = 0;

    if (useLiveData && depthData?.bids?.length && depthData?.asks?.length) {
      const topN = 20;
      const bidDepth = depthData.bids[Math.min(topN - 1, depthData.bids.length - 1)]?.cum_size ?? 0;
      const askDepth = depthData.asks[Math.min(topN - 1, depthData.asks.length - 1)]?.cum_size ?? 0;
      const total = bidDepth + askDepth;
      liquidityValue = total > 0 ? (bidDepth - askDepth) / total : 0;
    } else if (snapshot?.bands?.[0]) {
      liquidityValue = Math.max(-1, Math.min(1, Number(snapshot.bands[0].weighted_obi) || 0));
    } else {
      return null;
    }

    if (useLiveData && live.trades?.length) {
      const recent = live.trades.slice(-100);
      let buyVol = 0;
      let sellVol = 0;
      recent.forEach((t) => {
        const s = Number(t.size) || 0;
        if ((t.side || "").toLowerCase() === "buy") buyVol += s;
        else sellVol += s;
      });
      const totalVol = buyVol + sellVol;
      aggressionValue = totalVol > 0 ? (buyVol - sellVol) / totalVol : 0;
    } else if (flow) {
      aggressionValue = Math.max(-1, Math.min(1, Number(flow.aggressive_imbalance) || 0));
    }

    if (useLiveData && depthData?.bids?.length && depthData?.asks?.length && live.trades?.length) {
      const totalDepth = (depthData.bids[depthData.bids.length - 1]?.cum_size ?? 0) + (depthData.asks[depthData.asks.length - 1]?.cum_size ?? 0);
      const recentVol = live.trades.slice(-100).reduce((acc, t) => acc + (Number(t.size) || 0), 0);
      const ratio = recentVol > 0 ? totalDepth / recentVol : 0;
      absorptionPercent = Math.min(100, Math.max(0, (1 / (1 + 1 / (ratio || 1))) * 100));
    } else if (flow) {
      const absorptionRatio = Number(flow.absorption_ratio) || 0;
      absorptionPercent = Math.min(100, Math.max(0, (1 / (1 + 1 / (absorptionRatio || 1))) * 100));
    }

    return {
      liquidityValue: Math.max(-1, Math.min(1, liquidityValue)),
      aggressionValue: Math.max(-1, Math.min(1, aggressionValue)),
      absorptionPercent,
    };
  }, [snapshot, flow, depthData, live.connected, live.trades]);
  
  const flowBars = useMemo(() => {
    if (!flowSeries?.points?.length) return [];
    return flowSeries.points.map((p) => ({ t: p.t, buy: p.buy, sell: -p.sell, delta: p.delta }));
  }, [flowSeries]);

  const cvdSeries = useMemo(() => {
    if (!flowSeries?.points?.length) return [];
    return flowSeries.points.map((p) => ({ t: p.t, cvd: p.cvd, close: p.close }));
  }, [flowSeries]);

  // Align volume and CVD to candle timestamps so they overlay correctly (same time axis as OHLC).
  const flowByMinute = useMemo(() => {
    if (!flowSeries?.points?.length) return new Map();
    const m = new Map();
    flowSeries.points.forEach((p) => {
      const t = p.t;
      if (t == null) return;
      const ms = typeof t === "number" ? (t < 1e12 ? t * 1000 : t) : new Date(t).getTime();
      const minuteKey = Math.floor(ms / 60000) * 60000;
      m.set(minuteKey, p);
    });
    return m;
  }, [flowSeries]);

  const candleFlowBars = useMemo(() => {
    if (!priceCandles.length) return [];
    const intervalMinutes = (PRICE_TIMEFRAME_INTERVAL[priceTimeframe] ?? 60) / 60;
    const bucketMs = intervalMinutes * 60 * 1000;
    return priceCandles.map((c) => {
      const ts = c.timestamp;
      const bucketStartMs = typeof ts === "number" ? (ts < 1e12 ? ts * 1000 : ts) : new Date(ts).getTime();
      const bucketStartMinute = Math.floor(bucketStartMs / 60000) * 60000;
      let buy = 0, sell = 0;
      for (let m = 0; m < intervalMinutes; m += 1) {
        const minuteKey = bucketStartMinute + m * 60000;
        const p = flowByMinute.get(minuteKey);
        if (p) {
          buy += Number(p.buy) || 0;
          sell += Math.abs(Number(p.sell)) || 0;
        }
      }
      return { t: ts, buy, sell: -sell, delta: buy - sell };
    });
  }, [priceCandles, flowByMinute, priceTimeframe]);

  const candleCvdSeries = useMemo(() => {
    if (!priceCandles.length) return [];
    const intervalMinutes = (PRICE_TIMEFRAME_INTERVAL[priceTimeframe] ?? 60) / 60;
    const points = flowSeries?.points ?? [];
    const cvdByMinute = new Map();
    points.forEach((p) => {
      const t = p.t;
      if (t == null) return;
      const ms = typeof t === "number" ? (t < 1e12 ? t * 1000 : t) : new Date(t).getTime();
      const minuteKey = Math.floor(ms / 60000) * 60000;
      cvdByMinute.set(minuteKey, Number(p.cvd));
    });
    return priceCandles.map((c) => {
      const ts = c.timestamp;
      const bucketStartMs = typeof ts === "number" ? (ts < 1e12 ? ts * 1000 : ts) : new Date(ts).getTime();
      const bucketStartMinute = Math.floor(bucketStartMs / 60000) * 60000;
      const lastMinuteInBucket = bucketStartMinute + (intervalMinutes - 1) * 60000;
      const cvd = cvdByMinute.get(lastMinuteInBucket) ?? cvdByMinute.get(bucketStartMinute) ?? 0;
      return { t: ts, cvd, close: c.close };
    });
  }, [priceCandles, flowSeries?.points, priceTimeframe]);

  const fundingSeries = useMemo(() => {
    if (!funding?.points?.length) return [];
    return funding.points.map((p) => ({ t: p.t, funding: p.funding_rate, momentum: p.momentum }));
  }, [funding]);

  const oiSeries = useMemo(() => {
    if (!openInterest?.points?.length) return [];
    return openInterest.points.map((p) => ({ t: p.t, oi: p.open_interest, delta: p.delta }));
  }, [openInterest]);

  const depthSeries = useMemo(() => {
    if (!depthData?.bids?.length || !depthData?.asks?.length) return [];

    const rows = [];
    depthData.bids.forEach((p) => rows.push({ price: p.price, bidCum: p.cum_size, askCum: null }));
    depthData.asks.forEach((p) => rows.push({ price: p.price, bidCum: null, askCum: p.cum_size }));

    rows.sort((a, b) => a.price - b.price);
    return rows;
  }, [depthData]);

  const heatmap = useMemo(() => {
    // Snapshot proxy: show cumulative depth intensity side-by-side.
    if (!depthData?.bids?.length || !depthData?.asks?.length) return null;

    const bidLevels = depthData.bids.slice(0, 40);
    const askLevels = depthData.asks.slice(0, 40);

    const bidMax = bidLevels[bidLevels.length - 1]?.cum_size || 1;
    const askMax = askLevels[askLevels.length - 1]?.cum_size || 1;

    const rows = [];
    for (let i = 0; i < Math.max(bidLevels.length, askLevels.length); i += 1) {
      const b = bidLevels[i];
      const a = askLevels[i];
      rows.push({
        idx: i,
        bidPrice: b?.price,
        bidVal: b ? b.cum_size / bidMax : 0,
        askPrice: a?.price,
        askVal: a ? a.cum_size / askMax : 0,
      });
    }
    return rows;
  }, [depthData]);

  function timeframeToRange() {
    if (preset === "15m") {
      const end = new Date();
      const start = new Date(end.getTime() - 15 * 60 * 1000);
      return { start, end };
    }
    if (preset === "60m") {
      const end = new Date();
      const start = new Date(end.getTime() - 60 * 60 * 1000);
      return { start, end };
    }
    if (preset === "4h") {
      const end = new Date();
      const start = new Date(end.getTime() - 4 * 60 * 60 * 1000);
      return { start, end };
    }

    // custom
    return {
      start: dateTimeLocalValueToDate(customStart, useUTC) ?? new Date(customStart),
      end: dateTimeLocalValueToDate(customEnd, useUTC) ?? new Date(customEnd),
    };
  }

  // Maximum history: compute start so we request as much as the backend allows (~100k bars).
  // Approx bars = days * (bars per day): 1m=1440, 5m=288, 15m=96, 1h=24, 4h=6, 1d=1, 1w=1/7.
  function priceRangeForRequest() {
    const end = new Date();
    const opt = PRICE_RANGE_OPTIONS.find((o) => o.value === priceRange) || PRICE_RANGE_OPTIONS[0];
    if (opt.days != null) {
      const start = new Date(end.getTime() - opt.days * 24 * 60 * 60 * 1000);
      return { start, end };
    }
    // "Max": timeframe-dependent cap to stay under PRICE_CANDLES_SOFT_CAP
    const barsPerDay = { "1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1, "1w": 1 / 7 }[priceTimeframe] || 1440;
    const maxBars = Math.min(PRICE_CANDLES_SOFT_CAP, 100000);
    const maxDays = Math.floor(maxBars / barsPerDay);
    const days = Math.min(maxDays, priceTimeframe === "1w" ? 365 * 5 : priceTimeframe === "1d" ? 365 * 5 : 365 * 2);
    const start = new Date(end.getTime() - days * 24 * 60 * 60 * 1000);
    return { start, end };
  }

  async function loadSymbols() {
    const res = await api.get("/bitmex/symbols");
    setSymbols(res.data);
  }

  async function loadPriceCandles() {
    setError(null);
    setPriceChartLoading(true);
    const { start, end } = priceRangeForRequest();
    try {
      const res = await api.get("/bitmex/candles", {
        params: { symbol, start: start.toISOString(), end: end.toISOString(), bin_size: priceTimeframe },
        timeout: 60000,
      });
      const raw = res.data ?? [];
      setPriceCandles(raw.length > PRICE_CANDLES_MAX ? raw.slice(-PRICE_CANDLES_MAX) : raw);
    } catch (e) {
      const msg = e?.response?.data?.detail ?? "Failed to load price history";
      setError(msg);
    } finally {
      setPriceChartLoading(false);
    }
    // Load historical flow in background so chart finishes quickly; volume/CVD appear when ready.
    const rangeDays = (end.getTime() - start.getTime()) / (24 * 60 * 60 * 1000);
    const effectiveFlowDays = Math.min(rangeDays, flowMaxDays);
    const flowStart =
      rangeDays > flowMaxDays
        ? new Date(end.getTime() - flowMaxDays * 24 * 60 * 60 * 1000)
        : start;
    if (process.env.NODE_ENV !== "production") {
      console.log("Flow request:", flowStart.toISOString(), "to", end.toISOString(), "days:", Math.round(effectiveFlowDays * 10) / 10, "chartRangeDays:", Math.round(rangeDays * 10) / 10);
    }
    api
      .get("/bitmex/flow/timeseries", {
        params: { symbol, start: flowStart.toISOString(), end: end.toISOString() },
        timeout: 90000,
      })
      .then((flowRes) => {
        const d = flowRes.data;
        if (!d) return;
        const points = d.points?.length
          ? (d.points.length > FLOW_SERIES_POINTS_MAX ? d.points.slice(-FLOW_SERIES_POINTS_MAX) : d.points)
          : [];
        setFlowSeries({ ...d, points });
      })
      .catch(() => {});

  }

  // Split responsibilities to reduce memory/CPU: structure (fast), slow (funding/OI), candles (price only).
  // Do not fetch flow/timeseries here: it would overwrite flowSeries with only last N minutes and zero out
  // volume/CVD on the price chart (which needs flow for the full chart range from loadPriceCandles).
  async function refreshStructure() {
    const range = timeframeToRange();
    const coreSettled = await Promise.allSettled([
      api.get("/bitmex/analytics/snapshot", { params: { symbol, depth, bands_bps: bands } }),
      api.get("/bitmex/analytics/flow", { params: { symbol, minutes: Math.min(flowMinutes, 60) } }),
      api.get("/bitmex/orderbook/depth", { params: { symbol, depth } }),
    ]);
    const [snapRes, flowRes, depthRes] = coreSettled;
    if (snapRes.status === "fulfilled" && !live.connected) setSnapshot(snapRes.value.data);
    if (flowRes.status === "fulfilled") setFlow(flowRes.value.data);
    if (depthRes.status === "fulfilled" && !live.connected) setDepthData(depthRes.value.data);
    const firstErr = coreSettled.find((r) => r.status === "rejected");
    if (firstErr) setError(firstErr?.reason?.response?.data?.detail || "Structure fetch failed");
  }

  async function refreshSlow() {
    const range = timeframeToRange();
    const advSettled = await Promise.allSettled([
      api.get("/bitmex/funding", { params: { symbol, start: range.start.toISOString(), end: range.end.toISOString() } }),
      api.get("/bitmex/open-interest", { params: { symbol, start: range.start.toISOString(), end: range.end.toISOString() } }),
    ]);
    if (advSettled[0].status === "fulfilled") {
      const d = advSettled[0].value.data;
      if (d?.points?.length > FUNDING_POINTS_MAX) setFunding({ ...d, points: d.points.slice(-FUNDING_POINTS_MAX) });
      else setFunding(d);
    }
    if (advSettled[1].status === "fulfilled") {
      const d = advSettled[1].value.data;
      if (d?.points?.length > OI_POINTS_MAX) setOpenInterest({ ...d, points: d.points.slice(-OI_POINTS_MAX) });
      else setOpenInterest(d);
    }
  }

  async function refreshAll() {
    setBusy(true);
    setError(null);
    try {
      await refreshStructure();
      await loadPriceCandles();
      await refreshSlow();
    } finally {
      setBusy(false);
      if (process.env.NODE_ENV !== "production") {
        console.debug("[Analytics memory]", "candles:", priceCandles.length, "flow:", flowSeries?.points?.length ?? 0);
      }
    }
  }

  useEffect(() => {
    loadSymbols().catch(() => {});
  }, []);

  // In local dev (React 18 StrictMode), mount effects run twice. Guard to avoid
  // double WebSocket connections while keeping production behavior unchanged.
  useEffect(() => {
    if (didConnectLiveRef.current) return;
    didConnectLiveRef.current = true;
    if (live?.connected) return;
    try {
      live?.connect?.();
    } catch (e) {
      console.error("analytics: live.connect failed", e);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-reconnect when disconnected: schedule one attempt after 5s, no spam.
  useEffect(() => {
    if (live.connected) {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      return;
    }

    if (!reconnectTimerRef.current) {
      reconnectTimerRef.current = setTimeout(() => {
        console.warn("[Live] Disconnected — attempting auto-reconnect");
        live.connect();
        reconnectTimerRef.current = null;
      }, 5000);
    }

    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };
  }, [live.connected, live]);

  // Initial load: guard against StrictMode double-mount so we don't double-fetch.
  useEffect(() => {
    if (didInitRef.current) return;
    didInitRef.current = true;
    refreshStructure().then(() => loadPriceCandles().catch(() => {})).then(() => refreshSlow().catch(() => {})).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll only structure + candles (not slow: funding/OI) to reduce memory/CPU.
  useEffect(() => {
    if (!pollingEnabled) return;
    const intervalMs = Math.max(2, Number(pollIntervalSec || 5)) * 1000;
    refreshStructure().then(() => loadPriceCandles().catch(() => {})).catch(() => {});
    const id = setInterval(() => {
      refreshStructure().then(() => loadPriceCandles().catch(() => {})).catch(() => {});
    }, intervalMs);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pollingEnabled, pollIntervalSec, symbol, depth, bands, flowMinutes, preset, customStart, customEnd]);

  // Reload candles when symbol, timeframe, range or flow history days change so chart reflects selection.
  useEffect(() => {
    loadPriceCandles().catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, priceTimeframe, priceRange, flowMaxDays]);

  // When user changes symbol, restart live feed so WS subscribes to the new instrument (chart + orderbook then match).
  const prevSymbolRef = useRef(symbol);
  useEffect(() => {
    if (prevSymbolRef.current === symbol) return;
    prevSymbolRef.current = symbol;
    if (!live.connected) return;
    live.restart(symbol).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol]);

  // Warn if CVD/flow and candle timestamps diverge (different sampling can break axis alignment).
  useEffect(() => {
    if (!flowSeries?.points?.length || !priceCandles.length) return;
    const lastCandleTs = priceCandles[priceCandles.length - 1]?.timestamp;
    const lastFlowT = flowSeries.points[flowSeries.points.length - 1]?.t;
    if (lastCandleTs == null || lastFlowT == null) return;
    const candleMs = typeof lastCandleTs === "number" ? (lastCandleTs < 1e12 ? lastCandleTs * 1000 : lastCandleTs) : new Date(lastCandleTs).getTime();
    const flowMs = typeof lastFlowT === "number" ? (lastFlowT < 1e12 ? lastFlowT * 1000 : lastFlowT) : new Date(lastFlowT).getTime();
    const diffMs = Math.abs(candleMs - flowMs);
    const intervalMs = (PRICE_TIMEFRAME_INTERVAL[priceTimeframe] ?? 60) * 1000;
    if (diffMs > intervalMs * 2) {
      console.warn("[Analytics] Candle and flow/CVD timestamps diverge; chart alignment may be off.", { lastCandleTs, lastFlowT, diffMs });
    }
  }, [flowSeries, priceCandles, priceTimeframe]);

  return (
    <div data-testid="analytics-page" className="space-y-6">
      {/* Page title and global help */}
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h1 data-testid="analytics-title" className="text-2xl sm:text-3xl font-semibold tracking-tight">
            Analytics
          </h1>
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" size="sm" className="rounded-full text-muted-foreground" data-testid="analytics-how-to-use-button">
                How to use this?
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-[420px] text-sm space-y-3" align="end">
              <p className="font-medium">What this page is for</p>
              <p>Analytics gives you live and historical context: order book structure, flow, funding and open interest. Use it to understand where liquidity sits, who is aggressive, and how the market is absorbing orders.</p>
              <p className="font-medium">Order book analytics</p>
              <p>These tools are good at showing <strong>structure</strong> (where size sits), <strong>context</strong> (bid vs ask imbalance, aggression), and <strong>absorption</strong> (flow vs price move). They are <strong>not</strong> direct buy/sell signals—they help you read the market, not trigger entries by themselves.</p>
              <p className="font-medium">Backtesting</p>
              <p>Metrics and signals you see here (e.g. liquidity imbalance, aggression, absorption) can be used as inputs on the <strong>Backtesting</strong> page to test strategies. Think of Analytics as the observation layer; Backtesting is where you turn that into rules and evaluate them.</p>
            </PopoverContent>
          </Popover>
        </div>
        <div data-testid="analytics-subtitle" className="text-sm text-muted-foreground -mt-1">
          Charts-first view: depth, flow, funding, OI. Instrument and range set below.
        </div>
      </div>

      {/* Instrument-first context bar: symbol, live status, price timeframe & range */}
      <div
        data-testid="analytics-context-bar"
        className="flex flex-wrap items-center gap-3 rounded-xl border bg-card/50 px-4 py-3"
      >
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground whitespace-nowrap">Symbol</span>
          <Select value={symbol} onValueChange={setSymbol}>
            <SelectTrigger data-testid="analytics-symbol-select" className="w-[140px] h-8 rounded-lg text-xs">
              <SelectValue placeholder="Select symbol" />
            </SelectTrigger>
            <SelectContent data-testid="analytics-symbol-select-content">
              {symbols.length ? (
                symbols.map((s) => (
                  <SelectItem data-testid={`analytics-symbol-option-${s.symbol}`} key={s.symbol} value={s.symbol}>
                    {s.symbol}
                  </SelectItem>
                ))
              ) : (
                <SelectItem data-testid="analytics-symbol-option-loading" value="XBTUSD">Loading…</SelectItem>
              )}
            </SelectContent>
          </Select>
          <ChartHelp title="Symbol">
            <p>Instrument you are analysing. All charts and order book data use this symbol. Typical: <strong>XBTUSD</strong>. Change when you want to analyse another perpetual or index.</p>
          </ChartHelp>
        </div>
        <Badge
          data-testid="live-connection-badge"
          className="rounded-full text-xs text-white"
          style={{
            backgroundColor: live.connected ? "#16a34a" : "#dc2626",
            borderColor: live.connected ? "#16a34a" : "#dc2626",
          }}
        >
          {live.connected ? "Live" : "Disconnected"}
        </Badge>
        <Badge data-testid="live-symbol-badge" variant="outline" className="rounded-full text-xs">
          WS: {live.status?.symbol || symbol}
        </Badge>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground whitespace-nowrap">Candle</span>
          <div role="tablist" className="inline-flex h-8 items-center rounded-lg bg-muted p-0.5 text-muted-foreground flex-wrap">
            {["1m", "5m", "15m", "1h", "4h", "1d", "1w"].map((tf) => (
              <button
                key={tf}
                type="button"
                role="tab"
                data-testid={`analytics-price-timeframe-${tf}`}
                aria-selected={priceTimeframe === tf}
                className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${priceTimeframe === tf ? "bg-background text-foreground shadow" : "hover:text-foreground"}`}
                onClick={() => setPriceTimeframe(tf)}
              >
                {tf}
              </button>
            ))}
          </div>
          <ChartHelp title="Price timeframe">
            <p>Candle size for the main price chart. <strong>1m</strong> = 1-minute bars; <strong>1d</strong> = daily. Typical: 1m or 5m for short-term, 1h/4h for context. Change when you need different zoom or want to align with a strategy timeframe.</p>
          </ChartHelp>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Range</span>
          <Select value={priceRange} onValueChange={setPriceRange}>
            <SelectTrigger className="w-[160px] h-8 rounded-lg text-xs" data-testid="analytics-price-range-select">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PRICE_RANGE_OPTIONS.map((o) => (
                <SelectItem key={o.value} value={o.value} data-testid={`analytics-price-range-${o.value}`}>
                  {o.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <ChartHelp title="Price range">
            <p>How much history the price chart shows. <strong>7d</strong> = last 7 days; <strong>Max</strong> = all available. Typical: 30d. Change when you need more or less context; larger ranges can take longer to load.</p>
          </ChartHelp>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground whitespace-nowrap">Timezone</span>
          <Select value={useUTC ? "utc" : "local"} onValueChange={onTimezoneChange}>
            <SelectTrigger data-testid="analytics-context-timezone-select" className="w-[100px] h-8 rounded-lg text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem data-testid="analytics-context-timezone-utc" value="utc">UTC</SelectItem>
              <SelectItem data-testid="analytics-context-timezone-local" value="local">Local</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Tabs and chart content */}
      <div className="w-full" data-testid="analytics-tabs">
        <div role="tablist" data-testid="analytics-tabs-list" className="inline-flex h-9 items-center justify-center rounded-lg bg-muted p-1 text-muted-foreground rounded-full">
          <button
            type="button"
            role="tab"
            data-testid="analytics-tab-price"
            data-state={selectedTab === "price" ? "active" : "inactive"}
            className={`inline-flex items-center justify-center whitespace-nowrap rounded-full px-3 py-1 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ${selectedTab === "price" ? "bg-background text-foreground shadow" : ""}`}
            onClick={() => setSelectedTab("price")}
          >
            Price
          </button>
          <button
            type="button"
            role="tab"
            data-testid="analytics-tab-orderbook"
            data-state={selectedTab === "orderbook" ? "active" : "inactive"}
            className={`inline-flex items-center justify-center whitespace-nowrap rounded-full px-3 py-1 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ${selectedTab === "orderbook" ? "bg-background text-foreground shadow" : ""}`}
            onClick={() => setSelectedTab("orderbook")}
          >
            Order book
          </button>
          <button
            type="button"
            role="tab"
            data-testid="analytics-tab-flow"
            data-state={selectedTab === "flow" ? "active" : "inactive"}
            className={`inline-flex items-center justify-center whitespace-nowrap rounded-full px-3 py-1 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ${selectedTab === "flow" ? "bg-background text-foreground shadow" : ""}`}
            onClick={() => setSelectedTab("flow")}
          >
            Flow & CVD
          </button>
          <button
            type="button"
            role="tab"
            data-testid="analytics-tab-funding"
            data-state={selectedTab === "funding" ? "active" : "inactive"}
            className={`inline-flex items-center justify-center whitespace-nowrap rounded-full px-3 py-1 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ${selectedTab === "funding" ? "bg-background text-foreground shadow" : ""}`}
            onClick={() => setSelectedTab("funding")}
          >
            Funding + OI
          </button>
        </div>

        {selectedTab === "price" && (
        <div data-testid="analytics-tabs-price-content" role="tabpanel" className="mt-4">
          <Card data-testid="analytics-price-card" className="rounded-2xl">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle data-testid="analytics-price-title" className="text-base">Price (OHLC)</CardTitle>
              <ChartHelp title="What this chart shows">
                <p>This is the main price chart: each candle shows Open, High, Low and Close for that period. Green candles mean price closed higher than it opened; red means it closed lower. Use it to see structure, support/resistance and trend. Zoom with the scroll wheel, pan by dragging.</p>
              </ChartHelp>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap items-center gap-2 mb-3">
                <label className="flex items-center gap-1.5 cursor-pointer text-xs text-muted-foreground hover:text-foreground">
                  <Checkbox
                    checked={showVolumeOverlay}
                    onCheckedChange={(v) => setShowVolumeOverlay(!!v)}
                    data-testid="analytics-price-overlay-volume"
                    aria-label="Overlay volume on price chart"
                  />
                  Volume
                  <ChartHelp title="Volume overlay">
                    <p>Shows buy/sell volume bars on the price chart (green = buy, red = sell). Typical: off for a clean chart; turn on when you want to see where volume clustered. Data comes from the same flow used for the chart range.</p>
                  </ChartHelp>
                </label>
                <label className="flex items-center gap-1.5 cursor-pointer text-xs text-muted-foreground hover:text-foreground">
                  <Checkbox
                    checked={showCvdOverlay}
                    onCheckedChange={(v) => setShowCvdOverlay(!!v)}
                    data-testid="analytics-price-overlay-cvd"
                    aria-label="Overlay CVD on price chart"
                  />
                  CVD
                  <ChartHelp title="CVD overlay">
                    <p>Cumulative Volume Delta: running sum of buy volume minus sell volume. Helps see whether buyers or sellers are in control over time. Typical: off; turn on with Volume when you want flow context on the same time axis as price.</p>
                  </ChartHelp>
                </label>
              </div>
              <div data-testid="analytics-price-chart" className="relative h-[380px]">
                <PriceChart
                  candles={priceCandles}
                  cvdSeries={showCvdOverlay ? candleCvdSeries : []}
                  flowBars={showVolumeOverlay ? candleFlowBars : []}
                  height={380}
                  isVisible={selectedTab === "price"}
                  livePrice={livePriceForChart}
                  intervalSeconds={PRICE_TIMEFRAME_INTERVAL[priceTimeframe] ?? 60}
                />
                {priceChartLoading && (
                  <div
                    data-testid="analytics-price-chart-loading"
                    className="absolute inset-0 z-10 flex items-center justify-center rounded-lg bg-background/80 backdrop-blur-[1px]"
                    aria-hidden="true"
                  >
                    <div className="h-10 w-10 rounded-full border-2 border-muted border-t-primary animate-spin" />
                  </div>
                )}
              </div>
              <div data-testid="analytics-price-hint" className="mt-2 text-xs text-muted-foreground">
                OHLC candlesticks. Use the Volume and CVD checkboxes to overlay them on the chart. Signals at time t use data ≤ t; backtest entries trigger on next candle open.
              </div>
              <div className="mt-1 text-xs text-muted-foreground" data-testid="analytics-price-timezone-label">
                Times in {useUTC ? "UTC" : "Local"}.
              </div>
              {(() => {
                const pr = priceRangeForRequest();
                const rangeDays = (pr.end.getTime() - pr.start.getTime()) / (24 * 60 * 60 * 1000);
                if (rangeDays > flowMaxDays) {
                  return (
                    <div className="mt-2 text-xs text-amber-500" data-testid="analytics-flow-limited-warning">
                      Volume/CVD data limited to {flowMaxDays} days due to data availability.
                    </div>
                  );
                }
                return null;
              })()}
            </CardContent>
          </Card>
        </div>
        )}

        {selectedTab === "orderbook" && (
        <div data-testid="analytics-tabs-orderbook-content" role="tabpanel" className="mt-4 space-y-4">
          <div className="rounded-xl border bg-card/50 p-4">
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" size="sm" className="rounded-full text-muted-foreground" data-testid="analytics-orderbook-help-trigger">
                  How to read order book analytics
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-[400px] text-sm space-y-3" align="start">
                <p className="font-medium">Depth chart</p>
                <p>Cumulative bid and ask size by price level. Steep steps = lots of liquidity there. Use it to spot support, resistance and where large orders sit.</p>
                <p className="font-medium">Heatmap</p>
                <p>Snapshot-style view of bid vs ask depth intensity. Darker = more size. Quick way to see where liquidity is stacked.</p>
                <p className="font-medium">OBI (Order Book Imbalance)</p>
                <p>Per-band imbalance: more bids than asks (positive) or more asks (negative). Tells you if the book is bid-heavy or ask-heavy near the touch.</p>
                <p className="font-medium">Composite</p>
                <p>Combined liquidity and aggression scores from OBI and flow. Gives a quick read of structure and absorption—context, not entry signals.</p>
                <p className="text-muted-foreground border-t pt-2 mt-2">
                  These are <strong>contextual tools</strong> to read the market. They do not by themselves tell you when to buy or sell.
                </p>
              </PopoverContent>
            </Popover>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card data-testid="analytics-depth-card" className="rounded-2xl">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle data-testid="analytics-depth-title" className="text-base">Depth chart</CardTitle>
                <ChartHelp title="Depth chart">
                  <p>Shows cumulative size of bids (green) and asks (purple) at each price level. Steep steps mean a lot of liquidity at that price. Traders use it to spot support and resistance and where large orders sit.</p>
                </ChartHelp>
              </CardHeader>
              <CardContent>
                <div data-testid="analytics-depth-chart" className="h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={depthSeries} margin={{ left: 8, right: 8, top: 10, bottom: 0 }} isAnimationActive={false}>
                      <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="4 6" />
                      <XAxis type="number" dataKey="price" domain={["dataMin", "dataMax"]} tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <Tooltip
                        contentStyle={TOOLTIP_CONTENT_STYLE}
                        formatter={(val, name) => [fmt(val, 0), name]}
                        labelFormatter={(lab) => `price ${fmt(lab, 1)}`}
                      />
                      <Line type="stepAfter" dataKey="bidCum" stroke="hsl(var(--chart-2))" strokeWidth={2} dot={false} />
                      <Line type="stepAfter" dataKey="askCum" stroke="hsl(var(--chart-3))" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div data-testid="analytics-depth-hint" className="mt-2 text-xs text-muted-foreground">
                  Green: bids cumulative size. Purple: asks cumulative size.
                </div>
              </CardContent>
            </Card>

            <Card data-testid="analytics-heatmap-card" className="rounded-2xl">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle data-testid="analytics-heatmap-title" className="text-base">Heatmap (snapshot proxy)</CardTitle>
                <ChartHelp title="Heatmap">
                  <p>Side-by-side view of bid vs ask depth intensity. Darker bars mean more size at that level. Quick way to see where liquidity is stacked without reading exact numbers.</p>
                </ChartHelp>
              </CardHeader>
              <CardContent>
                <div data-testid="analytics-heatmap" className="rounded-xl border overflow-hidden">
                  <div className="grid grid-cols-2">
                    <div className="p-3 text-xs text-muted-foreground border-r">Bids</div>
                    <div className="p-3 text-xs text-muted-foreground">Asks</div>
                  </div>
                  <div className="divide-y">
                    {(heatmap || []).slice(0, 30).map((r) => (
                      <div data-testid={`analytics-heatmap-row-${r.idx}`} key={r.idx} className="grid grid-cols-2">
                        <div className="flex items-center justify-between gap-3 p-3 border-r">
                          <div data-testid={`analytics-heatmap-bid-price-${r.idx}`} className="text-xs text-muted-foreground">
                            {r.bidPrice ? fmt(r.bidPrice, 1) : "—"}
                          </div>
                          <div
                            data-testid={`analytics-heatmap-bid-cell-${r.idx}`}
                            className="h-4 w-24 rounded-full border"
                            style={{ background: colorForHeat(r.bidVal), borderColor: "hsl(var(--border))" }}
                          />
                        </div>
                        <div className="flex items-center justify-between gap-3 p-3">
                          <div data-testid={`analytics-heatmap-ask-price-${r.idx}`} className="text-xs text-muted-foreground">
                            {r.askPrice ? fmt(r.askPrice, 1) : "—"}
                          </div>
                          <div
                            data-testid={`analytics-heatmap-ask-cell-${r.idx}`}
                            className="h-4 w-24 rounded-full border"
                            style={{ background: colorForHeat(r.askVal), borderColor: "hsl(var(--border))" }}
                          />
                        </div>
                      </div>
                    ))}
                    {!heatmap ? (
                      <div data-testid="analytics-heatmap-loading" className="p-4 text-sm text-muted-foreground">
                        Loading…
                      </div>
                    ) : null}
                  </div>
                </div>
                <div data-testid="analytics-heatmap-hint" className="mt-2 text-xs text-muted-foreground">
                  MVP note: this is a snapshot-style heat proxy (true time heatmap comes next).
                </div>
              </CardContent>
            </Card>
          </div>

          <Card data-testid="analytics-obi-card" className="rounded-2xl">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle data-testid="analytics-obi-title" className="text-base">Order book imbalance by band</CardTitle>
              <ChartHelp title="OBI by band">
                <p>Order Book Imbalance (OBI) in price bands around the mid. Positive means more bid depth than ask in that band; negative means more asks. Weighted OBI emphasizes size. Helps see if the book is bid-heavy or ask-heavy near the touch.</p>
              </ChartHelp>
            </CardHeader>
            <CardContent className="space-y-3">
              {snapshot?.bands?.length ? (
                snapshot.bands.map((b) => (
                  <div data-testid={`analytics-obi-row-${b.band_bps}`} key={b.band_bps} className="rounded-xl border bg-card/60 p-3">
                    <div className="flex items-center justify-between gap-4 flex-wrap">
                      <div className="flex items-center gap-2">
                        <Badge data-testid={`analytics-obi-band-badge-${b.band_bps}`} variant="secondary" className="rounded-full">
                          {b.band_bps} bps
                        </Badge>
                        <div data-testid={`analytics-obi-obi-${b.band_bps}`} className="text-sm font-medium">
                          OBI {fmt(b.obi, 3)}
                        </div>
                        <div data-testid={`analytics-obi-wobi-${b.band_bps}`} className="text-sm text-muted-foreground">
                          (weighted {fmt(b.weighted_obi, 3)})
                        </div>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        <span data-testid={`analytics-obi-depth-${b.band_bps}`}>bid {fmt(b.bid_depth, 0)} / ask {fmt(b.ask_depth, 0)}</span>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div data-testid="analytics-obi-loading" className="text-sm text-muted-foreground">
                  Loading…
                </div>
              )}
            </CardContent>
          </Card>
        </div>
        )}

        {selectedTab === "flow" && (
        <div data-testid="analytics-tabs-flow-content" role="tabpanel" className="mt-4 space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card data-testid="analytics-cvd-card" className="rounded-2xl">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle data-testid="analytics-cvd-title" className="text-base">CVD + price overlay</CardTitle>
                <ChartHelp title="CVD (Cumulative Volume Delta)">
                  <p>This chart shows cumulative volume delta (CVD), which helps identify whether aggressive buyers or sellers are in control. It adds up buy volume minus sell volume over time. Rising CVD with rising price suggests buying pressure; falling CVD with falling price suggests selling pressure.</p>
                </ChartHelp>
              </CardHeader>
              <CardContent>
                <div data-testid="analytics-cvd-chart" className="h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={cvdSeries} margin={{ left: 8, right: 8, top: 10, bottom: 24 }} isAnimationActive={false}>
                      <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="4 6" />
                      <XAxis
                        dataKey="t"
                        tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                        tickFormatter={(t) => formatTime(t, useUTC)}
                        axisLine={{ stroke: "hsl(var(--border))" }}
                        tickLine={{ stroke: "hsl(var(--border))" }}
                      />
                      <YAxis
                        yAxisId="left"
                        orientation="left"
                        tick={{ fill: "hsl(var(--chart-5))", fontSize: 11 }}
                        axisLine={{ stroke: "hsl(var(--border))" }}
                        tickLine={{ stroke: "hsl(var(--border))" }}
                        allowDataOverflow
                        width={48}
                        label={{ value: "CVD", angle: -90, position: "insideLeft", fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
                      />
                      <YAxis
                        yAxisId="right"
                        orientation="right"
                        tick={{ fill: "hsl(var(--chart-4))", fontSize: 11 }}
                        axisLine={{ stroke: "hsl(var(--border))" }}
                        tickLine={{ stroke: "hsl(var(--border))" }}
                        allowDataOverflow
                        width={56}
                        label={{ value: "Price", angle: 90, position: "insideRight", fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
                      />
                      <Tooltip
                        contentStyle={TOOLTIP_CONTENT_STYLE}
                        labelFormatter={(label) => formatTime(label, useUTC)}
                        formatter={(val, name) => [fmt(val, name === "close" ? 2 : 0), name === "close" ? "Price" : "CVD"]}
                      />
                      <Line yAxisId="left" type="monotone" dataKey="cvd" name="CVD" stroke="hsl(var(--chart-5))" strokeWidth={2} dot={false} />
                      <Line yAxisId="right" type="monotone" dataKey="close" name="Price" stroke="hsl(var(--chart-4))" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div data-testid="analytics-cvd-hint" className="mt-2 text-xs text-muted-foreground">
                  CVD from per-minute buy-sell delta (computed from trades). Price is 1m close. All series use candle-close timestamps; signals at t use data ≤ t.
                </div>
                <div className="mt-1 text-xs text-muted-foreground" data-testid="analytics-cvd-timezone-label">
                  Times in {useUTC ? "UTC" : "Local"}.
                </div>
              </CardContent>
            </Card>

            <Card data-testid="analytics-aggr-bars-card" className="rounded-2xl">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle data-testid="analytics-aggr-bars-title" className="text-base">Aggressive buy/sell bars</CardTitle>
                <ChartHelp title="Aggressive buy/sell volume">
                  <p>Each bar shows how much volume traded on the buy side (green) vs sell side (red) in that period. Traders use this to see who is more aggressive and whether the market is absorbing or driving price. Sell bars are shown negative so you can compare both sides at a glance.</p>
                </ChartHelp>
              </CardHeader>
              <CardContent>
                <div data-testid="analytics-aggr-bars-chart" className="h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={flowBars} margin={{ left: 8, right: 8, top: 10, bottom: 0 }} isAnimationActive={false}>
                      <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="4 6" />
                      <XAxis dataKey="t" hide />
                      <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <Tooltip
                        contentStyle={TOOLTIP_CONTENT_STYLE}
                        labelFormatter={() => ""}
                        formatter={(val, name) => [fmt(val, 0), name]}
                      />
                      <Bar dataKey="buy" fill="hsl(var(--chart-2))" radius={[8, 8, 0, 0]} />
                      <Bar dataKey="sell" fill="hsl(var(--chart-3))" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div data-testid="analytics-aggr-bars-hint" className="mt-2 text-xs text-muted-foreground">
                  Sell bars are plotted negative to visually separate sides.
                </div>
                <div className="mt-1 text-xs text-muted-foreground" data-testid="analytics-aggr-bars-timezone-label">
                  Times in {useUTC ? "UTC" : "Local"}.
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
        )}

        {selectedTab === "funding" && (
        <div data-testid="analytics-tabs-funding-content" role="tabpanel" className="mt-4 space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card data-testid="analytics-funding-card" className="rounded-2xl">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle data-testid="analytics-funding-title" className="text-base">Funding rate + momentum</CardTitle>
                <ChartHelp title="Funding rate">
                  <p>Funding is the periodic payment between longs and shorts in perpetual futures. Positive rate means longs pay shorts (often when price is above fair value). Momentum is the change between funding points. Traders watch for extremes and shifts to gauge sentiment and mean reversion.</p>
                </ChartHelp>
              </CardHeader>
              <CardContent>
                <div data-testid="analytics-funding-chart" className="h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={fundingSeries} margin={{ left: 8, right: 8, top: 10, bottom: 0 }} isAnimationActive={false}>
                      <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="4 6" />
                      <XAxis dataKey="t" hide />
                      <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <Tooltip
                        contentStyle={TOOLTIP_CONTENT_STYLE}
                        labelFormatter={() => ""}
                        formatter={(val, name) => [fmt(val, 6), name]}
                      />
                      <Line type="monotone" dataKey="funding" stroke="hsl(var(--chart-1))" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="momentum" stroke="hsl(var(--chart-4))" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div data-testid="analytics-funding-hint" className="mt-2 text-xs text-muted-foreground">
                  Momentum is simple delta between consecutive funding points.
                </div>
                <div className="mt-1 text-xs text-muted-foreground" data-testid="analytics-funding-timezone-label">
                  Times in {useUTC ? "UTC" : "Local"}.
                </div>
              </CardContent>
            </Card>

            <Card data-testid="analytics-oi-card" className="rounded-2xl">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle data-testid="analytics-oi-title" className="text-base">Open interest + delta</CardTitle>
                <ChartHelp title="Open interest (OI)">
                  <p>Open interest is the total number of open contracts. When OI rises with price, new money is coming in; when OI falls with price, positions are closing. The delta line shows how much OI changed between points. Traders use it to confirm or question a move.</p>
                </ChartHelp>
              </CardHeader>
              <CardContent>
                <div data-testid="analytics-oi-chart" className="h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={oiSeries} margin={{ left: 8, right: 8, top: 10, bottom: 0 }} isAnimationActive={false}>
                      <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="4 6" />
                      <XAxis dataKey="t" hide />
                      <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <Tooltip
                        contentStyle={TOOLTIP_CONTENT_STYLE}
                        labelFormatter={() => ""}
                        formatter={(val, name) => [fmt(val, 0), name]}
                      />
                      <Line type="monotone" dataKey="oi" stroke="hsl(var(--chart-5))" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="delta" stroke="hsl(var(--chart-2))" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div data-testid="analytics-oi-hint" className="mt-2 text-xs text-muted-foreground">
                  Delta is change in open interest between points.
                </div>
                <div className="mt-1 text-xs text-muted-foreground" data-testid="analytics-oi-timezone-label">
                  Times in {useUTC ? "UTC" : "Local"}.
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
        )}

      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge data-testid="live-rate-badge" variant="secondary" className="rounded-full text-xs">
              {fmt(live.liveMsgsPerSec || 0, 1)} msg/s
            </Badge>
            <span data-testid="live-last-message" className="text-xs text-muted-foreground">
              {live.lastMessageAt ? `Last: ${Math.round((Date.now() - live.lastMessageAt) / 1000)}s ago` : "Waiting for live…"}
            </span>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <Button
              data-testid="live-restart-button"
              variant="outline"
              className="rounded-full"
              onClick={() => live.restart(symbol)}
            >
              Restart live feed
            </Button>
            <Button
              data-testid="live-connect-button"
              variant="outline"
              className="rounded-full"
              onClick={() => live.connect()}
            >
              Reconnect
            </Button>

            <Popover open={showPollingMenu} onOpenChange={setShowPollingMenu}>
  <PopoverTrigger asChild>
    <Button variant={pollingEnabled ? "secondary" : "outline"} className="rounded-full">
      Polling: {pollingEnabled ? "ON" : "OFF"}
    </Button>
  </PopoverTrigger>
  <PopoverContent className="w-64 space-y-3 text-sm">
    <div className="flex justify-between items-center">
      <span>Enable polling</span>
      <Button
        size="sm"
        variant={pollingEnabled ? "secondary" : "outline"}
        onClick={() => setPollingEnabled(v => !v)}
      >
        {pollingEnabled ? "ON" : "OFF"}
      </Button>
    </div>

    <div className="space-y-1">
      <Label>Interval (sec)</Label>
      <Input
        type="number"
        min={2}
        max={60}
        value={pollIntervalSec}
        onChange={(e) => setPollIntervalSec(Number(e.target.value))}
      />
    </div>

    <p className="text-xs text-muted-foreground">
      Polling uses REST snapshots instead of live WebSocket data.
      Useful if live feeds are unstable.
    </p>
  </PopoverContent>
</Popover>


            <Button data-testid="analytics-refresh-button" className="rounded-full" onClick={refreshAll} disabled={busy}>
              {busy ? "Refreshing…" : "Refresh now"}
            </Button>
          </div>
        </div>
      </div>

      <Card data-testid="analytics-controls-card" className="rounded-2xl">
        <CardHeader>
          <CardTitle className="text-base">Controls</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div className="space-y-2">
              <div className="flex items-center gap-1.5">
                <Label data-testid="analytics-depth-label" htmlFor="depth">L2 Depth</Label>
                <ChartHelp title="L2 Depth">
                  <p>Number of order book levels (bids and asks) used for depth charts and OBI. Higher = more structure and context, slower updates. Typical: 25–50. Increase when you need to see deeper into the book.</p>
                </ChartHelp>
              </div>
              <Input
                data-testid="analytics-depth-input"
                id="depth"
                type="number"
                value={depth}
                min={10}
                max={200}
                onChange={(e) => setDepth(Number(e.target.value))}
                className="rounded-xl"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-1.5">
                <Label data-testid="analytics-bands-label" htmlFor="bands">Bands (bps)</Label>
                <ChartHelp title="Bands (bps)">
                  <p>Price bands around mid in basis points (e.g. 10, 25, 100). Used to compute OBI and liquidity metrics per band. Typical: 10,25,100. Change to focus on tighter or wider zones around the touch.</p>
                </ChartHelp>
              </div>
              <Input
                data-testid="analytics-bands-input"
                id="bands"
                value={bands}
                onChange={(e) => setBands(e.target.value)}
                placeholder="10,25,100"
                className="rounded-xl"
              />
              <div data-testid="analytics-bands-hint" className="text-xs text-muted-foreground">
                Example: 10,25,100 (basis points around mid)
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-1.5">
                <Label data-testid="analytics-flow-minutes-label" htmlFor="flowMinutes">Flow (min)</Label>
                <ChartHelp title="Flow (min)">
                  <p>Window in minutes for aggressive buy/sell and CVD calculations (Flow card, Regime, Composite). Typical: 60. Increase for longer context; decrease for more reactive, short-term flow.</p>
                </ChartHelp>
              </div>
              <Input
                data-testid="analytics-flow-minutes-input"
                id="flowMinutes"
                type="number"
                value={flowMinutes}
                min={5}
                max={240}
                onChange={(e) => setFlowMinutes(Number(e.target.value))}
                className="rounded-xl"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-1.5">
                <Label data-testid="analytics-flow-max-days-label" htmlFor="flowMaxDays">Flow history (days)</Label>
                <ChartHelp title="Flow history (days)">
                  <p>Maximum days of volume/CVD data to load for the price chart. Volume and CVD overlays use this range. Increase for longer history (e.g. 365); lower values load faster.</p>
                </ChartHelp>
              </div>
              <Input
                data-testid="analytics-flow-max-days-input"
                id="flowMaxDays"
                type="number"
                value={flowMaxDays}
                min={7}
                max={730}
                onChange={(e) => setFlowMaxDays(Math.min(730, Math.max(7, Number(e.target.value) || 7)))}
                className="rounded-xl"
              />
              <div data-testid="analytics-flow-max-days-hint" className="text-xs text-muted-foreground">
                Volume/CVD range on price chart (7–730 days).
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-1.5">
                <Clock className="h-4 w-4 text-muted-foreground" aria-hidden />
                <Label data-testid="analytics-time-display-label">Timezone</Label>
                <ChartHelp title="Timezone">
                  <p>Show timestamps in UTC or local time in chart axes and tooltips.</p>
                </ChartHelp>
              </div>
              <Select value={useUTC ? "utc" : "local"} onValueChange={onTimezoneChange}>
                <SelectTrigger data-testid="analytics-time-display-select" className="rounded-xl">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem data-testid="analytics-time-utc" value="utc">UTC</SelectItem>
                  <SelectItem data-testid="analytics-time-local" value="local">Local</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <Separator />

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
            <div className="space-y-2">
              <div className="flex items-center gap-1.5">
                <Label data-testid="analytics-timeframe-label">Timeframe</Label>
                <ChartHelp title="Timeframe (structure)">
                  <p>Preset range for funding, OI and some flow context (separate from the price chart range). Typical: Last 60m or 4h. Use Custom to set exact start/end for historical analysis.</p>
                </ChartHelp>
              </div>
              <Select value={preset} onValueChange={setPreset}>
                <SelectTrigger data-testid="analytics-timeframe-select" className="rounded-xl">
                  <SelectValue placeholder="Select timeframe" />
                </SelectTrigger>
                <SelectContent data-testid="analytics-timeframe-select-content">
                  <SelectItem data-testid="analytics-timeframe-15m" value="15m">Last 15m</SelectItem>
                  <SelectItem data-testid="analytics-timeframe-60m" value="60m">Last 60m</SelectItem>
                  <SelectItem data-testid="analytics-timeframe-4h" value="4h">Last 4h</SelectItem>
                  <SelectItem data-testid="analytics-timeframe-custom" value="custom">Custom</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label data-testid="analytics-custom-start-label" htmlFor="customStart">Custom start</Label>
              <Input
                data-testid="analytics-custom-start-input"
                id="customStart"
                type="datetime-local"
                value={customStart}
                onChange={(e) => setCustomStart(e.target.value)}
                className="rounded-xl"
                disabled={preset !== "custom"}
              />
              <div className="text-xs text-muted-foreground" data-testid="analytics-custom-start-timezone-hint">
                Times in {useUTC ? "UTC" : "Local"}.
              </div>
            </div>

            <div className="space-y-2">
              <Label data-testid="analytics-custom-end-label" htmlFor="customEnd">Custom end</Label>
              <Input
                data-testid="analytics-custom-end-input"
                id="customEnd"
                type="datetime-local"
                value={customEnd}
                onChange={(e) => setCustomEnd(e.target.value)}
                className="rounded-xl"
                disabled={preset !== "custom"}
              />
              <div className="text-xs text-muted-foreground" data-testid="analytics-custom-end-timezone-hint">
                Times in {useUTC ? "UTC" : "Local"}.
              </div>
            </div>
          </div>

          {error ? (
            <div data-testid="analytics-error" className="text-sm text-destructive">
              {error}
            </div>
          ) : null}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card data-testid="analytics-market-card" className="rounded-2xl">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle data-testid="analytics-market-title" className="text-base">Market</CardTitle>
            <ChartHelp title="Market (top of book)">
              <p>Live best bid, best ask, mid and spread from the order book. When the live feed is connected, these update in real time. The latest OHLC candle (open, high, low, close) is also shown for the current timeframe. Use this to see where price is trading and how tight the spread is.</p>
            </ChartHelp>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <div data-testid="analytics-market-bestbid-label" className="text-sm text-muted-foreground">Best bid</div>
              <div data-testid="analytics-market-bestbid" className="text-sm font-medium">{fmt(snapshot?.best_bid, 1)}</div>
            </div>
            <div className="flex items-center justify-between">
              <div data-testid="analytics-market-bestask-label" className="text-sm text-muted-foreground">Best ask</div>
              <div data-testid="analytics-market-bestask" className="text-sm font-medium">{fmt(snapshot?.best_ask, 1)}</div>
            </div>
            <Separator />
            <div className="flex items-center justify-between">
              <div data-testid="analytics-market-mid-label" className="text-sm text-muted-foreground">Mid</div>
              <div data-testid="analytics-market-mid" className="text-sm font-medium">{fmt(snapshot?.mid, 1)}</div>
            </div>
            <div className="flex items-center justify-between">
              <div data-testid="analytics-market-spread-label" className="text-sm text-muted-foreground">Spread</div>
              <div data-testid="analytics-market-spread" className="text-sm font-medium">{fmt(snapshot?.spread, 1)}</div>
            </div>
            <Separator />

              {latestCandle ? (
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Open</span>
                    <span>{fmt(latestCandle.open, 1)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">High</span>
                    <span>{fmt(latestCandle.high, 1)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Low</span>
                    <span>{fmt(latestCandle.low, 1)}</span>
                  </div>
                  <div className="flex justify-between font-medium">
                    <span>Close</span>
                    <span>{fmt(latestCandle.close, 1)}</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Candle time: {formatTime(latestCandle.timestamp, useUTC)}
                    <span className="text-muted-foreground text-xs ml-1">({useUTC ? "UTC" : "Local"})</span>
                  </div>
                </div>
              ) : null}

            <div className="pt-2">
              <Badge data-testid="analytics-market-ts" variant="secondary" className="rounded-full">
                {snapshot?.ts ? (
                  <>
                    Updated {formatTime(snapshot.ts, useUTC)}
                    <span className="text-muted-foreground text-xs ml-1">({useUTC ? "UTC" : "Local"})</span>
                  </>
                ) : (
                  "Loading…"
                )}
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="rounded-2xl">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-base">Regime metrics</CardTitle>
            <ChartHelp title="Regime metrics">
              <p>Derived values from the order book and recent flow (not trading signals).</p>
              <p><strong>Liquidity</strong>: Weighted OBI of the top band; left = ask-dominant, right = bid-dominant.</p>
              <p><strong>Aggression</strong>: Buy vs sell volume imbalance; left = seller-driven, right = buyer-driven.</p>
              <p><strong>Absorption</strong>: Book depth relative to traded flow (0–100%); higher means more absorption.</p>
            </ChartHelp>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            {regimeMetrics ? (
              <>
                <div data-testid="analytics-regime-liquidity">
                  <SpectrumGauge
                    title="Liquidity"
                    value={regimeMetrics.liquidityValue}
                    leftLabel="Ask-dominant"
                    rightLabel="Bid-dominant"
                  />
                </div>
                <div data-testid="analytics-regime-aggression">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-medium text-muted-foreground">Aggression</span>
                    <VelocityArrows value={regimeMetrics.aggressionValue} />
                  </div>
                  <SpectrumGauge
                    value={regimeMetrics.aggressionValue}
                    leftLabel="Seller-driven"
                    rightLabel="Buyer-driven"
                  />
                </div>
                <div data-testid="analytics-regime-absorption">
                  <PercentGauge
                    title="Absorption"
                    value={regimeMetrics.absorptionPercent}
                  />
                </div>
              </>
            ) : (
              <div className="text-muted-foreground">Waiting for data…</div>
            )}
          </CardContent>
        </Card>


        <Card data-testid="analytics-flow-card" className="rounded-2xl">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle data-testid="analytics-flow-title" className="text-base">Flow (short window)</CardTitle>
            <ChartHelp title="Flow (short window)">
              <p>Aggregated buy vs sell volume and CVD over the short window (e.g. 60 minutes). <strong>Aggressive imbalance</strong> is buy volume minus sell volume normalized; positive means more buying. <strong>CVD</strong> is cumulative volume delta. <strong>Absorption ratio</strong> compares book depth to traded flow—high values suggest the market is absorbing orders rather than trending. Use with Regime metrics for context.</p>
            </ChartHelp>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <div data-testid="analytics-flow-buy-label" className="text-sm text-muted-foreground">Buy volume</div>
              <div data-testid="analytics-flow-buy" className="text-sm font-medium">{fmt(flow?.buy_volume, 0)}</div>
            </div>
            <div className="flex items-center justify-between">
              <div data-testid="analytics-flow-sell-label" className="text-sm text-muted-foreground">Sell volume</div>
              <div data-testid="analytics-flow-sell" className="text-sm font-medium">{fmt(flow?.sell_volume, 0)}</div>
            </div>
            <Separator />
            <div className="flex items-center justify-between">
              <div data-testid="analytics-flow-imbalance-label" className="text-sm text-muted-foreground">Aggressive imbalance</div>
              <div data-testid="analytics-flow-imbalance" className="text-sm font-medium">{fmt(flow?.aggressive_imbalance, 3)}</div>
            </div>
            <div className="flex items-center justify-between">
              <div data-testid="analytics-flow-cvd-label" className="text-sm text-muted-foreground">CVD</div>
              <div data-testid="analytics-flow-cvd" className="text-sm font-medium">{fmt(flow?.cvd, 0)}</div>
            </div>
            <Separator />
            <div className="flex items-center justify-between">
              <div data-testid="analytics-flow-absorption-label" className="text-sm text-muted-foreground">Absorption ratio</div>
              <div data-testid="analytics-flow-absorption" className="text-sm font-medium">{fmt(flow?.absorption_ratio, 2)}</div>
            </div>
          </CardContent>
        </Card>

        <Card data-testid="analytics-composite-card" className="rounded-2xl">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle data-testid="analytics-composite-title" className="text-base">Composite</CardTitle>
            <ChartHelp title="Composite">
              <p><strong>Liquidity defense</strong>: Weighted OBI of the top band expressed as a 0–100% score; higher means more bid-side depth near the touch.</p>
              <p><strong>Aggression efficiency</strong>: A 0–100% measure derived from absorption ratio; higher means flow is being absorbed more efficiently. These composites combine order book and flow to give a quick read of structure and aggression.</p>
            </ChartHelp>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <div data-testid="analytics-composite-liquidity-label" className="text-sm text-muted-foreground">Liquidity defense</div>
              <div data-testid="analytics-composite-liquidity" className="text-sm font-medium">
                {snapshot?.bands?.length ? fmt(((snapshot.bands[0].weighted_obi + 1) / 2) * 100, 0) : "—"}%
              </div>
            </div>
            <div data-testid="analytics-composite-liquidity-hint" className="text-xs text-muted-foreground">
              Derived from weighted OBI (top band). Higher means bid-side depth is stronger.
            </div>

            <Separator />

            <div className="flex items-center justify-between">
              <div data-testid="analytics-composite-aggr-label" className="text-sm text-muted-foreground">Aggression efficiency</div>
              <div data-testid="analytics-composite-aggr" className="text-sm font-medium">
                {flow ? fmt(Math.min(100, Math.max(0, (1 / (1 + 1 / (flow.absorption_ratio || 1))) * 100)), 0) : "—"}%
              </div>
            </div>
            <div data-testid="analytics-composite-aggr-hint" className="text-xs text-muted-foreground">
              Uses absorption proxy (aggressive volume per unit price move). Higher = more absorption.
            </div>

            <div className="pt-2 flex flex-wrap items-center gap-2">
              <Badge data-testid="analytics-composite-note" variant="secondary" className="rounded-full">
                No buy/sell labels — only context.
              </Badge>
            </div>
            <div className="rounded-lg border border-dashed bg-muted/30 p-3 mt-3 text-xs text-muted-foreground" data-testid="analytics-backtest-bridge">
              <p className="font-medium text-foreground mb-1">Signals derived here are used in Backtesting</p>
              <p>On the Backtesting page you can use these metrics as inputs to conditions:</p>
              <ul className="list-disc list-inside mt-1 space-y-0.5">
                <li><strong>Liquidity imbalance</strong> — from OBI / weighted OBI by band</li>
                <li><strong>Aggression imbalance</strong> — buy vs sell volume in the flow window</li>
                <li><strong>Absorption score</strong> — flow absorbed vs price move</li>
              </ul>
              <p className="mt-2">This card is informational only; editing and backtest setup happen on the Backtest page.</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Floating timezone indicator */}
      <button
        type="button"
        onClick={() => onTimezoneChange(useUTC ? "local" : "utc")}
        className="fixed bottom-6 right-6 z-50 transition-all duration-200 hover:scale-105 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-full"
        data-testid="analytics-floating-timezone"
        aria-label={`Timezone: ${useUTC ? "UTC" : "Local"}. Click to switch.`}
      >
        <Badge variant="secondary" className="rounded-full h-9 px-3 gap-1.5 text-xs font-medium shadow-md border bg-card/95 backdrop-blur">
          <Clock className="h-3.5 w-3.5 transition-opacity" aria-hidden />
          <span>{useUTC ? "UTC" : "Local"}</span>
        </Badge>
      </button>
    </div>
    </div>
  );
}
