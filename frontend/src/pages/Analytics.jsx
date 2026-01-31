import React, { useEffect, useMemo, useState } from "react";
import { api } from "@/api/client";
import { useLive } from "@/hooks/use-live";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";

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
  const live = useLive();
  const [symbols, setSymbols] = useState([]);
  const [symbol, setSymbol] = useState("XBTUSD");

  // shared controls
  const [depth, setDepth] = useState(50);
  const [bands, setBands] = useState("10,25,100");

  // timeframes
  const [flowMinutes, setFlowMinutes] = useState(60);
  const [liqMinutes, setLiqMinutes] = useState(60);
  const [preset, setPreset] = useState("60m");

  const [customStart, setCustomStart] = useState(() => {
    const d = new Date(Date.now() - 1000 * 60 * 60);
    return d.toISOString().slice(0, 16);
  });
  const [customEnd, setCustomEnd] = useState(() => {
    const d = new Date();
    return d.toISOString().slice(0, 16);
  });

  const [snapshot, setSnapshot] = useState(null);
  const [flow, setFlow] = useState(null);
  const [priceCandles, setPriceCandles] = useState([]);

  const [depthData, setDepthData] = useState(null);
  const [flowSeries, setFlowSeries] = useState(null);

  const [funding, setFunding] = useState(null);
  const [openInterest, setOpenInterest] = useState(null);
  const [liquidations, setLiquidations] = useState(null);

  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const priceSeries = useMemo(() => {
    return priceCandles.map((c) => ({ t: c.timestamp, close: c.close }));
  }, [priceCandles]);

  const flowBars = useMemo(() => {
    if (!flowSeries?.points?.length) return [];
    return flowSeries.points.map((p) => ({ t: p.t, buy: p.buy, sell: -p.sell, delta: p.delta }));
  }, [flowSeries]);

  const cvdSeries = useMemo(() => {
    if (!flowSeries?.points?.length) return [];
    return flowSeries.points.map((p) => ({ t: p.t, cvd: p.cvd, close: p.close }));
  }, [flowSeries]);

  const fundingSeries = useMemo(() => {
    if (!funding?.points?.length) return [];
    return funding.points.map((p) => ({ t: p.t, funding: p.funding_rate, momentum: p.momentum }));
  }, [funding]);

  const oiSeries = useMemo(() => {
    if (!openInterest?.points?.length) return [];
    return openInterest.points.map((p) => ({ t: p.t, oi: p.open_interest, delta: p.delta }));
  }, [openInterest]);

  const liqSeries = useMemo(() => {
    if (!liquidations?.points?.length) return [];
    return liquidations.points.map((p) => ({ t: p.t, price: p.price, size: p.size, side: p.side }));
  }, [liquidations]);

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
    return { start: new Date(customStart), end: new Date(customEnd) };
  }

  async function loadSymbols() {
    const res = await api.get("/bitmex/symbols");
    setSymbols(res.data);
  }

  async function refreshAll() {
    setBusy(true);
    setError(null);

    const range = timeframeToRange();

    try {
      const coreSettled = await Promise.allSettled([
        api.get("/bitmex/analytics/snapshot", { params: { symbol, depth, bands_bps: bands } }),
        api.get("/bitmex/analytics/flow", { params: { symbol, minutes: Math.min(flowMinutes, 60) } }),
        api.get("/bitmex/orderbook/depth", { params: { symbol, depth } }),
        api.get("/bitmex/flow/timeseries", { params: { symbol, minutes: flowMinutes } }),
      ]);

      const [snapRes, flowRes, depthRes, flowSeriesRes] = coreSettled;

      // If live feed has an orderbook snapshot, use it for the top-of-page market values.
      if (live.orderbook) {
        setSnapshot((prev) => {
          const bands = prev?.bands || [];
          return {
            symbol: live.orderbook.symbol,
            ts: live.orderbook.ts,
            best_bid: live.orderbook.best_bid,
            best_ask: live.orderbook.best_ask,
            mid: live.orderbook.mid,
            spread: live.orderbook.spread,
            bands,
          };
        });
      }

      if (snapRes.status === "fulfilled") setSnapshot(snapRes.value.data);
      if (flowRes.status === "fulfilled") setFlow(flowRes.value.data);
      if (depthRes.status === "fulfilled") setDepthData(depthRes.value.data);
      if (flowSeriesRes.status === "fulfilled") setFlowSeries(flowSeriesRes.value.data);

      const candleSettled = await Promise.allSettled([
        api.get("/bitmex/candles", {
          params: { symbol, start: range.start.toISOString(), end: range.end.toISOString() },
        }),
      ]);
      if (candleSettled[0].status === "fulfilled") {
        setPriceCandles(candleSettled[0].value.data.slice(-600));
      }

      const advSettled = await Promise.allSettled([
        api.get("/bitmex/funding", { params: { symbol, start: range.start.toISOString(), end: range.end.toISOString() } }),
        api.get("/bitmex/open-interest", { params: { symbol, start: range.start.toISOString(), end: range.end.toISOString() } }),
        api.get("/bitmex/liquidations", { params: { symbol, minutes: liqMinutes } }),
      ]);

      if (advSettled[0].status === "fulfilled") setFunding(advSettled[0].value.data);
      if (advSettled[1].status === "fulfilled") setOpenInterest(advSettled[1].value.data);
      if (advSettled[2].status === "fulfilled") setLiquidations(advSettled[2].value.data);

      const firstErr = [...coreSettled, ...candleSettled, ...advSettled].find((r) => r.status === "rejected");
      if (firstErr) {
        const msg = firstErr?.reason?.response?.data?.detail || "One or more data panels failed to load";
        setError(msg);
      }
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    loadSymbols().catch(() => {});
  }, []);

  useEffect(() => {
    refreshAll().catch(() => {});
    const id = setInterval(() => {
      refreshAll().catch(() => {});
    }, 5000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, depth, bands, flowMinutes, liqMinutes, preset, customStart, customEnd]);

  return (
    <div data-testid="analytics-page" className="space-y-6">
      <div className="flex flex-col gap-2">
        <div className="flex items-end justify-between gap-4 flex-wrap">
          <div>
            <h1 data-testid="analytics-title" className="text-3xl sm:text-4xl font-semibold tracking-tight">
              Analytics
            </h1>
            <div data-testid="analytics-subtitle" className="text-sm text-muted-foreground mt-1">
              Charts-first view: depth, heat, flow, funding, OI, liquidations.
            </div>
            <div className="mt-2 flex items-center gap-2 flex-wrap">
              <Badge
                data-testid="live-connection-badge"
                variant={live.connected ? "secondary" : "destructive"}
                className="rounded-full"
              >
                Live feed: {live.connected ? "Connected" : "Disconnected"}
              </Badge>
              <Badge data-testid="live-symbol-badge" variant="outline" className="rounded-full">
                WS symbol: {live.status?.symbol || symbol}
              </Badge>
              <div data-testid="live-last-message" className="text-xs text-muted-foreground">
                {live.lastMessageAt ? `Last live message: ${Math.round((Date.now() - live.lastMessageAt) / 1000)}s ago` : "Waiting for live messages…"}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-wrap justify-end">
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

            <Button data-testid="analytics-refresh-button" className="rounded-full" onClick={refreshAll} disabled={busy}>
              {busy ? "Refreshing…" : "Refresh (polling)"}
            </Button>
          </div>
        </div>
      </div>

      <Card data-testid="analytics-controls-card" className="rounded-2xl">
        <CardHeader>
          <CardTitle data-testid="analytics-controls-title" className="text-base">Controls</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div className="space-y-2">
              <Label data-testid="analytics-symbol-label">Symbol</Label>
              <Select value={symbol} onValueChange={setSymbol}>
                <SelectTrigger data-testid="analytics-symbol-select" className="rounded-xl">
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
                    <SelectItem data-testid="analytics-symbol-option-loading" value="XBTUSD">
                      Loading…
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label data-testid="analytics-depth-label" htmlFor="depth">
                L2 Depth
              </Label>
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
              <Label data-testid="analytics-bands-label" htmlFor="bands">
                Bands (bps)
              </Label>
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
              <Label data-testid="analytics-flow-minutes-label" htmlFor="flowMinutes">
                Flow (min)
              </Label>
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
              <Label data-testid="analytics-liq-minutes-label" htmlFor="liqMinutes">
                Liquidations (min)
              </Label>
              <Input
                data-testid="analytics-liq-minutes-input"
                id="liqMinutes"
                type="number"
                value={liqMinutes}
                min={5}
                max={240}
                onChange={(e) => setLiqMinutes(Number(e.target.value))}
                className="rounded-xl"
              />
            </div>
          </div>

          <Separator />

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
            <div className="space-y-2">
              <Label data-testid="analytics-timeframe-label">Timeframe</Label>
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
          <CardHeader>
            <CardTitle data-testid="analytics-market-title" className="text-base">Market</CardTitle>
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
            <div className="pt-2">
              <Badge data-testid="analytics-market-ts" variant="secondary" className="rounded-full">
                {snapshot?.ts ? `Updated ${snapshot.ts}` : "Loading…"}
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card data-testid="analytics-flow-card" className="rounded-2xl">
          <CardHeader>
            <CardTitle data-testid="analytics-flow-title" className="text-base">Flow (short window)</CardTitle>
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
          <CardHeader>
            <CardTitle data-testid="analytics-composite-title" className="text-base">Composite</CardTitle>
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

            <div className="pt-2">
              <Badge data-testid="analytics-composite-note" variant="secondary" className="rounded-full">
                No buy/sell labels — only context.
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="orderbook" className="w-full" data-testid="analytics-tabs">
        <TabsList data-testid="analytics-tabs-list" className="rounded-full">
          <TabsTrigger data-testid="analytics-tab-orderbook" value="orderbook" className="rounded-full">
            Order book
          </TabsTrigger>
          <TabsTrigger data-testid="analytics-tab-flow" value="flow" className="rounded-full">
            Flow
          </TabsTrigger>
          <TabsTrigger data-testid="analytics-tab-funding" value="funding" className="rounded-full">
            Funding + OI + Liquidations
          </TabsTrigger>
          <TabsTrigger data-testid="analytics-tab-price" value="price" className="rounded-full">
            Price
          </TabsTrigger>
        </TabsList>

        <TabsContent data-testid="analytics-tabs-orderbook-content" value="orderbook" className="mt-4 space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card data-testid="analytics-depth-card" className="rounded-2xl">
              <CardHeader>
                <CardTitle data-testid="analytics-depth-title" className="text-base">Depth chart</CardTitle>
              </CardHeader>
              <CardContent>
                <div data-testid="analytics-depth-chart" className="h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={depthSeries} margin={{ left: 8, right: 8, top: 10, bottom: 0 }}>
                      <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="4 6" />
                      <XAxis type="number" dataKey="price" domain={["dataMin", "dataMax"]} tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{ background: "hsl(var(--popover))", borderRadius: 12, border: "1px solid hsl(var(--border))" }}
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
              <CardHeader>
                <CardTitle data-testid="analytics-heatmap-title" className="text-base">Heatmap (snapshot proxy)</CardTitle>
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
            <CardHeader>
              <CardTitle data-testid="analytics-obi-title" className="text-base">Order book imbalance by band</CardTitle>
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
        </TabsContent>

        <TabsContent data-testid="analytics-tabs-flow-content" value="flow" className="mt-4 space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card data-testid="analytics-cvd-card" className="rounded-2xl">
              <CardHeader>
                <CardTitle data-testid="analytics-cvd-title" className="text-base">CVD + price overlay</CardTitle>
              </CardHeader>
              <CardContent>
                <div data-testid="analytics-cvd-chart" className="h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={cvdSeries} margin={{ left: 8, right: 8, top: 10, bottom: 0 }}>
                      <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="4 6" />
                      <XAxis dataKey="t" hide />
                      <YAxis yAxisId="left" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <YAxis yAxisId="right" orientation="right" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{ background: "hsl(var(--popover))", borderRadius: 12, border: "1px solid hsl(var(--border))" }}
                        formatter={(val, name) => [fmt(val, name === "close" ? 2 : 0), name]}
                        labelFormatter={() => ""}
                      />
                      <Line yAxisId="left" type="monotone" dataKey="cvd" stroke="hsl(var(--chart-5))" strokeWidth={2} dot={false} />
                      <Line yAxisId="right" type="monotone" dataKey="close" stroke="hsl(var(--chart-4))" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div data-testid="analytics-cvd-hint" className="mt-2 text-xs text-muted-foreground">
                  CVD from per-minute buy-sell delta (computed from trades). Price is 1m close.
                </div>
              </CardContent>
            </Card>

            <Card data-testid="analytics-aggr-bars-card" className="rounded-2xl">
              <CardHeader>
                <CardTitle data-testid="analytics-aggr-bars-title" className="text-base">Aggressive buy/sell bars</CardTitle>
              </CardHeader>
              <CardContent>
                <div data-testid="analytics-aggr-bars-chart" className="h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={flowBars} margin={{ left: 8, right: 8, top: 10, bottom: 0 }}>
                      <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="4 6" />
                      <XAxis dataKey="t" hide />
                      <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{ background: "hsl(var(--popover))", borderRadius: 12, border: "1px solid hsl(var(--border))" }}
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
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent data-testid="analytics-tabs-funding-content" value="funding" className="mt-4 space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card data-testid="analytics-funding-card" className="rounded-2xl">
              <CardHeader>
                <CardTitle data-testid="analytics-funding-title" className="text-base">Funding rate + momentum</CardTitle>
              </CardHeader>
              <CardContent>
                <div data-testid="analytics-funding-chart" className="h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={fundingSeries} margin={{ left: 8, right: 8, top: 10, bottom: 0 }}>
                      <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="4 6" />
                      <XAxis dataKey="t" hide />
                      <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{ background: "hsl(var(--popover))", borderRadius: 12, border: "1px solid hsl(var(--border))" }}
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
              </CardContent>
            </Card>

            <Card data-testid="analytics-oi-card" className="rounded-2xl">
              <CardHeader>
                <CardTitle data-testid="analytics-oi-title" className="text-base">Open interest + delta</CardTitle>
              </CardHeader>
              <CardContent>
                <div data-testid="analytics-oi-chart" className="h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={oiSeries} margin={{ left: 8, right: 8, top: 10, bottom: 0 }}>
                      <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="4 6" />
                      <XAxis dataKey="t" hide />
                      <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{ background: "hsl(var(--popover))", borderRadius: 12, border: "1px solid hsl(var(--border))" }}
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
              </CardContent>
            </Card>
          </div>

          <Card data-testid="analytics-liq-card" className="rounded-2xl">
            <CardHeader>
              <CardTitle data-testid="analytics-liq-title" className="text-base">Liquidations (recent) — cluster proxy</CardTitle>
            </CardHeader>
            <CardContent>
              <div data-testid="analytics-liq-chart" className="h-[320px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ left: 8, right: 8, top: 10, bottom: 0 }}>
                    <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="4 6" />
                    <XAxis dataKey="t" hide />
                    <YAxis dataKey="price" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{ background: "hsl(var(--popover))", borderRadius: 12, border: "1px solid hsl(var(--border))" }}
                      labelFormatter={() => ""}
                      formatter={(val, name) => [fmt(val, name === "price" ? 1 : 0), name]}
                    />
                    <Scatter data={liqSeries} fill="hsl(var(--chart-4))" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <div data-testid="analytics-liq-hint" className="mt-2 text-xs text-muted-foreground">
                Each dot is a liquidation print. Treat as a probabilistic stress / cluster proxy.
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent data-testid="analytics-tabs-price-content" value="price" className="mt-4">
          <Card data-testid="analytics-price-card" className="rounded-2xl">
            <CardHeader>
              <CardTitle data-testid="analytics-price-title" className="text-base">Price (1m close)</CardTitle>
            </CardHeader>
            <CardContent>
              <div data-testid="analytics-price-chart" className="h-[320px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={priceSeries} margin={{ left: 6, right: 6, top: 10, bottom: 0 }}>
                    <defs>
                      <linearGradient id="tmxPrice" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--chart-2))" stopOpacity={0.35} />
                        <stop offset="95%" stopColor="hsl(var(--chart-2))" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <XAxis hide dataKey="t" />
                    <YAxis hide domain={["dataMin", "dataMax"]} />
                    <Tooltip
                      contentStyle={{ background: "hsl(var(--popover))", borderRadius: 12, border: "1px solid hsl(var(--border))" }}
                      labelFormatter={() => ""}
                      formatter={(val) => [fmt(val, 2), "close"]}
                    />
                    <Area type="monotone" dataKey="close" stroke="hsl(var(--chart-2))" strokeWidth={2} fill="url(#tmxPrice)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div data-testid="analytics-price-hint" className="mt-2 text-xs text-muted-foreground">
                Timeframe follows the selector (15m/60m/4h/custom).
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
