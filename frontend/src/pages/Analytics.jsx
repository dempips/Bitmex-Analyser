import React, { useEffect, useMemo, useState } from "react";
import { api } from "@/api/client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

function fmt(n, digits = 2) {
  if (n === null || n === undefined) return "—";
  if (Number.isNaN(Number(n))) return "—";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: digits });
}

export default function Analytics() {
  const [symbols, setSymbols] = useState([]);
  const [symbol, setSymbol] = useState("XBTUSD");
  const [depth, setDepth] = useState(50);
  const [bands, setBands] = useState("10,25,100");
  const [minutes, setMinutes] = useState(5);

  const [snapshot, setSnapshot] = useState(null);
  const [flow, setFlow] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const [miniCandles, setMiniCandles] = useState([]);

  const chartData = useMemo(() => {
    return miniCandles.map((c) => ({ t: c.timestamp, close: c.close }));
  }, [miniCandles]);

  async function loadSymbols() {
    const res = await api.get("/bitmex/symbols");
    setSymbols(res.data);
  }

  async function refreshAll() {
    setBusy(true);
    setError(null);
    try {
      const [snapRes, flowRes] = await Promise.all([
        api.get("/bitmex/analytics/snapshot", { params: { symbol, depth, bands_bps: bands } }),
        api.get("/bitmex/analytics/flow", { params: { symbol, minutes } }),
      ]);
      setSnapshot(snapRes.data);
      setFlow(flowRes.data);

      const end = new Date();
      const start = new Date(end.getTime() - 60 * 60 * 1000);
      const candlesRes = await api.get("/bitmex/candles", {
        params: { symbol, start: start.toISOString(), end: end.toISOString() },
      });
      setMiniCandles(candlesRes.data.slice(-200));
    } catch (e) {
      setError(e?.response?.data?.detail || "Failed to load analytics");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    loadSymbols().catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    refreshAll().catch(() => {});
    const id = setInterval(() => {
      refreshAll().catch(() => {});
    }, 5000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, depth, bands, minutes]);

  return (
    <div data-testid="analytics-page" className="space-y-6">
      <div className="flex flex-col gap-2">
        <div className="flex items-end justify-between gap-4 flex-wrap">
          <div>
            <h1 data-testid="analytics-title" className="text-3xl sm:text-4xl font-semibold tracking-tight">
              Analytics
            </h1>
            <div data-testid="analytics-subtitle" className="text-sm text-muted-foreground mt-1">
              Snapshot + flow computed from BitMEX public REST. Polling every 5 seconds.
            </div>
          </div>
          <Button data-testid="analytics-refresh-button" className="rounded-full" onClick={refreshAll} disabled={busy}>
            {busy ? "Refreshing…" : "Refresh now"}
          </Button>
        </div>
      </div>

      <Card data-testid="analytics-controls-card" className="rounded-2xl">
        <CardHeader>
          <CardTitle data-testid="analytics-controls-title" className="text-base">Controls</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
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
              <Label data-testid="analytics-flow-minutes-label" htmlFor="minutes">
                Flow window (min)
              </Label>
              <Input
                data-testid="analytics-flow-minutes-input"
                id="minutes"
                type="number"
                value={minutes}
                min={1}
                max={60}
                onChange={(e) => setMinutes(Number(e.target.value))}
                className="rounded-xl"
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
          </CardContent>
        </Card>

        <Card data-testid="analytics-flow-card" className="rounded-2xl">
          <CardHeader>
            <CardTitle data-testid="analytics-flow-title" className="text-base">Trade flow (window)</CardTitle>
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
              <div data-testid="analytics-flow-pricechange-label" className="text-sm text-muted-foreground">Price change</div>
              <div data-testid="analytics-flow-pricechange" className="text-sm font-medium">{fmt(flow?.price_change, 2)}</div>
            </div>
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

      <Tabs defaultValue="obi" className="w-full" data-testid="analytics-tabs">
        <TabsList data-testid="analytics-tabs-list" className="rounded-full">
          <TabsTrigger data-testid="analytics-tab-obi" value="obi" className="rounded-full">
            OBI bands
          </TabsTrigger>
          <TabsTrigger data-testid="analytics-tab-price" value="price" className="rounded-full">
            Price (last 60m)
          </TabsTrigger>
        </TabsList>

        <TabsContent data-testid="analytics-tabs-obi-content" value="obi" className="mt-4">
          <Card data-testid="analytics-obi-card" className="rounded-2xl">
            <CardHeader>
              <CardTitle data-testid="analytics-obi-title" className="text-base">Order book imbalance by band</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {snapshot?.bands?.length ? (
                snapshot.bands.map((b) => (
                  <div
                    data-testid={`analytics-obi-row-${b.band_bps}`}
                    key={b.band_bps}
                    className="rounded-xl border bg-card/60 p-3"
                  >
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

        <TabsContent data-testid="analytics-tabs-price-content" value="price" className="mt-4">
          <Card data-testid="analytics-price-card" className="rounded-2xl">
            <CardHeader>
              <CardTitle data-testid="analytics-price-title" className="text-base">Close price (approx)</CardTitle>
            </CardHeader>
            <CardContent>
              <div data-testid="analytics-price-chart" className="h-[260px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData} margin={{ left: 6, right: 6, top: 10, bottom: 0 }}>
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
                This is a quick 1m candle pull over the last hour.
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
