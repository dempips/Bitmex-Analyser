import React, { useEffect, useMemo, useState } from "react";
import { api } from "@/api/client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PriceChart } from "@/components/PriceChart";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import {
  convertDateTimeLocalValue,
  dateTimeLocalValueToDate,
  dateToDateTimeLocalValue,
  formatTime,
  getStoredTimezone,
  setStoredTimezone,
} from "@/utils/time";

function fmt(n, digits = 2) {
  if (n === null || n === undefined) return "—";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: digits });
}

const metricOptions = [
  { value: "close", label: "Close" },
  { value: "return_1", label: "Return (1 bar)" },
  { value: "sma", label: "SMA" },
  { value: "ema", label: "EMA" },
  { value: "volatility", label: "Volatility (std of returns)" },
  { value: "liquidity_imbalance", label: "Liquidity imbalance (OBI)" },
  { value: "aggression_imbalance", label: "Aggression imbalance" },
  { value: "absorption_score", label: "Absorption score" },
];

const operators = [">", ">=", "<", "<=", "=="];
const periods = [10, 20, 50];

function ConditionEditor({ idx, value, onChange, onRemove, testPrefix }) {
  const showPeriod = value.metric === "sma" || value.metric === "ema" || value.metric === "volatility";
  const isSignalMetric = ["liquidity_imbalance", "aggression_imbalance", "absorption_score"].includes(value.metric);

  return (
    <div data-testid={`${testPrefix}-condition-${idx}`} className="rounded-2xl border bg-card/60 p-3">
      <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-end">
        <div className="md:col-span-4 space-y-2">
          <Label data-testid={`${testPrefix}-metric-label-${idx}`}>Metric</Label>
          <Select value={value.metric} onValueChange={(v) => onChange({ ...value, metric: v, period: showPeriod ? value.period : null })}>
            <SelectTrigger data-testid={`${testPrefix}-metric-select-${idx}`} className="rounded-xl">
              <SelectValue placeholder="Metric" />
            </SelectTrigger>
            <SelectContent data-testid={`${testPrefix}-metric-select-content-${idx}`}>
              {metricOptions.map((m) => (
                <SelectItem data-testid={`${testPrefix}-metric-option-${idx}-${m.value}`} key={m.value} value={m.value}>
                  {m.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="md:col-span-2 space-y-2">
          <Label data-testid={`${testPrefix}-operator-label-${idx}`}>Op</Label>
          <Select value={value.operator} onValueChange={(v) => onChange({ ...value, operator: v })}>
            <SelectTrigger data-testid={`${testPrefix}-operator-select-${idx}`} className="rounded-xl">
              <SelectValue placeholder="Op" />
            </SelectTrigger>
            <SelectContent data-testid={`${testPrefix}-operator-select-content-${idx}`}>
              {operators.map((op) => (
                <SelectItem data-testid={`${testPrefix}-operator-option-${idx}-${op}`} key={op} value={op}>
                  {op}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {showPeriod ? (
          <div className="md:col-span-2 space-y-2">
            <Label data-testid={`${testPrefix}-period-label-${idx}`}>Period</Label>
            <Select value={String(value.period || 10)} onValueChange={(v) => onChange({ ...value, period: Number(v) })}>
              <SelectTrigger data-testid={`${testPrefix}-period-select-${idx}`} className="rounded-xl">
                <SelectValue placeholder="Period" />
              </SelectTrigger>
              <SelectContent data-testid={`${testPrefix}-period-select-content-${idx}`}>
                {periods.map((p) => (
                  <SelectItem data-testid={`${testPrefix}-period-option-${idx}-${p}`} key={p} value={String(p)}>
                    {p}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        ) : (
          <div className="md:col-span-2" />
        )}

        <div className="md:col-span-3 space-y-2">
          <Label data-testid={`${testPrefix}-value-label-${idx}`}>Value</Label>
          <Input
            data-testid={`${testPrefix}-value-input-${idx}`}
            type="number"
            value={value.value}
            onChange={(e) => onChange({ ...value, value: Number(e.target.value) })}
            className="rounded-xl"
          />
          <div data-testid={`${testPrefix}-value-hint-${idx}`} className="text-xs text-muted-foreground">
            {isSignalMetric ? "OBI/aggression: -1 to +1; absorption: 0 to 100." : "Tip: returns are decimals (e.g., 0.002 = +0.2%)."}
          </div>
        </div>

        <div className="md:col-span-1 flex md:justify-end">
          <Button data-testid={`${testPrefix}-remove-condition-${idx}`} type="button" variant="outline" className="rounded-full" onClick={onRemove}>
            Remove
          </Button>
        </div>
      </div>
    </div>
  );
}

function computeDrawdownSeries(equityCurve) {
  let peak = -1e18;
  return equityCurve.map((p) => {
    const eq = Number(p.equity);
    peak = Math.max(peak, eq);
    const dd = peak > 0 ? (peak - eq) / peak : 0;
    return { t: p.t, drawdown: -dd * 100 };
  });
}

export default function Backtest() {
  const initialTz = getStoredTimezone("utc");
  const [useUTC, setUseUTC] = useState(initialTz === "utc");
  const [symbols, setSymbols] = useState([]);
  const [symbol, setSymbol] = useState("XBTUSD");

  const [name, setName] = useState("OBI Momentum (MVP)");
  const [feeBps, setFeeBps] = useState(7.5);
  const [slipBps, setSlipBps] = useState(2.0);

  const [entry, setEntry] = useState([{ metric: "ema", period: 20, operator: ">", value: 0 }]);
  const [exit, setExit] = useState([{ metric: "return_1", operator: "<", value: -0.002, period: null }]);

  const [start, setStart] = useState(() => {
    const d = new Date(Date.now() - 1000 * 60 * 60 * 24 * 2);
    return dateToDateTimeLocalValue(d, initialTz === "utc");
  });
  const [end, setEnd] = useState(() => {
    const d = new Date();
    return dateToDateTimeLocalValue(d, initialTz === "utc");
  });

  const [risk, setRisk] = useState({ stop_loss_pct: undefined, take_profit_pct: undefined, max_hold_bars: undefined });
  const [run, setRun] = useState(null);
  const [backtestCandles, setBacktestCandles] = useState([]);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  // Persist timezone preference across reloads.
  useEffect(() => {
    setStoredTimezone(useUTC ? "utc" : "local");
  }, [useUTC]);

  function onTimezoneChange(next) {
    const nextUseUTC = next === "utc";
    // Convert existing datetime-local inputs to preserve the same instant.
    setStart((v) => convertDateTimeLocalValue(v, useUTC, nextUseUTC));
    setEnd((v) => convertDateTimeLocalValue(v, useUTC, nextUseUTC));
    setUseUTC(nextUseUTC);
  }

  useEffect(() => {
    if (!run?.symbol || !run?.start || !run?.end) return;
    api
      .get("/bitmex/candles", { params: { symbol: run.symbol, start: run.start, end: run.end, bin_size: "1m" } })
      .then((res) => setBacktestCandles(res.data ?? []))
      .catch(() => setBacktestCandles([]));
  }, [run?.symbol, run?.start, run?.end]);

  const equitySeries = useMemo(() => {
    if (!run?.equity_curve?.length) return [];
    return run.equity_curve.map((p) => ({ t: p.t, equity: p.equity, price: p.price }));
  }, [run]);

  const ddSeries = useMemo(() => {
    if (!run?.equity_curve?.length) return [];
    return computeDrawdownSeries(run.equity_curve);
  }, [run]);

  const tradeMarkers = useMemo(() => {
    if (!run?.trades?.length || !run?.equity_curve?.length) return [];
    const byT = new Map(run.equity_curve.map((p) => [p.t, p.price]));
    const points = [];
    run.trades.forEach((t, i) => {
      const ep = byT.get(t.entry_time);
      const xp = byT.get(t.exit_time);
      if (ep) points.push({ t: t.entry_time, price: ep, kind: "entry", i });
      if (xp) points.push({ t: t.exit_time, price: xp, kind: "exit", i });
    });
    return points;
  }, [run]);

  const entryExitMarkersForChart = useMemo(() => {
    if (!run?.trades?.length) return [];
    const out = [];
    run.trades.forEach((t) => {
      out.push({ t: t.entry_time, kind: "entry" });
      out.push({ t: t.exit_time, kind: "exit" });
    });
    return out;
  }, [run?.trades]);

  async function loadSymbols() {
    const res = await api.get("/bitmex/symbols");
    setSymbols(res.data);
  }

  useEffect(() => {
    loadSymbols().catch(() => {});
  }, []);

  function addCondition(setter) {
    setter((prev) => [...prev, { metric: "close", operator: ">", value: 0, period: null }]);
  }

  async function runBacktest() {
    setBusy(true);
    setError(null);
    setRun(null);
    try {
      const startDate = dateTimeLocalValueToDate(start, useUTC);
      const endDate = dateTimeLocalValueToDate(end, useUTC);
      const payload = {
        symbol,
        start: (startDate ?? new Date(start)).toISOString(),
        end: (endDate ?? new Date(end)).toISOString(),
        strategy: {
          name,
          symbol,
          entry_conditions: entry,
          exit_conditions: exit,
          fee_bps: Number(feeBps),
          slippage_bps: Number(slipBps),
        },
        initial_capital: 10000,
        risk: Object.keys(risk).some((k) => risk[k] != null) ? risk : undefined,
      };
      const res = await api.post("/backtests/run", payload);
      setRun(res.data);
    } catch (e) {
      setError(e?.response?.data?.detail || "Backtest failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div data-testid="backtest-page" className="space-y-6">
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 data-testid="backtest-title" className="text-3xl sm:text-4xl font-semibold tracking-tight">Backtest</h1>
          <div data-testid="backtest-subtitle" className="text-sm text-muted-foreground mt-1">
            Rule-based, 1-minute candles. Long-only MVP.
          </div>
        </div>
        <Button data-testid="backtest-run-button" className="rounded-full" onClick={runBacktest} disabled={busy}>
          {busy ? "Running…" : "Run backtest"}
        </Button>
      </div>

      <Card data-testid="backtest-config-card" className="rounded-2xl">
        <CardHeader>
          <CardTitle data-testid="backtest-config-title" className="text-base">Strategy configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="space-y-2 md:col-span-2">
              <Label data-testid="backtest-name-label" htmlFor="name">Name</Label>
              <Input data-testid="backtest-name-input" id="name" value={name} onChange={(e) => setName(e.target.value)} className="rounded-xl" />
            </div>

            <div className="space-y-2">
              <Label data-testid="backtest-symbol-label">Symbol</Label>
              <Select value={symbol} onValueChange={setSymbol}>
                <SelectTrigger data-testid="backtest-symbol-select" className="rounded-xl">
                  <SelectValue placeholder="Select symbol" />
                </SelectTrigger>
                <SelectContent data-testid="backtest-symbol-select-content">
                  {symbols.length ? (
                    symbols.map((s) => (
                      <SelectItem data-testid={`backtest-symbol-option-${s.symbol}`} key={s.symbol} value={s.symbol}>
                        {s.symbol}
                      </SelectItem>
                    ))
                  ) : (
                    <SelectItem data-testid="backtest-symbol-option-loading" value="XBTUSD">Loading…</SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label data-testid="backtest-fee-label" htmlFor="fee">Fee (bps)</Label>
              <Input
                data-testid="backtest-fee-input"
                id="fee"
                type="number"
                value={feeBps}
                onChange={(e) => setFeeBps(Number(e.target.value))}
                className="rounded-xl"
              />
              <div data-testid="backtest-fee-hint" className="text-xs text-muted-foreground">
                Default approximates taker fee (can be tuned).
              </div>
            </div>

            <div className="space-y-2">
              <Label data-testid="backtest-slippage-label" htmlFor="slip">Slippage (bps)</Label>
              <Input
                data-testid="backtest-slippage-input"
                id="slip"
                type="number"
                value={slipBps}
                onChange={(e) => setSlipBps(Number(e.target.value))}
                className="rounded-xl"
              />
            </div>

            <div className="space-y-2">
              <Label data-testid="backtest-start-label" htmlFor="start">Start</Label>
              <Input
                data-testid="backtest-start-input"
                id="start"
                type="datetime-local"
                value={start}
                onChange={(e) => setStart(e.target.value)}
                className="rounded-xl"
              />
            </div>

            <div className="space-y-2">
              <Label data-testid="backtest-end-label" htmlFor="end">End</Label>
              <Input
                data-testid="backtest-end-input"
                id="end"
                type="datetime-local"
                value={end}
                onChange={(e) => setEnd(e.target.value)}
                className="rounded-xl"
              />
            </div>

            <div className="space-y-2">
              <Label data-testid="backtest-timezone-label">Timezone</Label>
              <Select value={useUTC ? "utc" : "local"} onValueChange={onTimezoneChange}>
                <SelectTrigger data-testid="backtest-timezone-select" className="rounded-xl">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem data-testid="backtest-timezone-utc" value="utc">UTC</SelectItem>
                  <SelectItem data-testid="backtest-timezone-local" value="local">Local</SelectItem>
                </SelectContent>
              </Select>
              <div data-testid="backtest-timezone-hint" className="text-xs text-muted-foreground">
                Start/end inputs are interpreted in this timezone.
              </div>
            </div>
          </div>

          <Separator />

          <div className="space-y-4">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div>
                <div data-testid="backtest-entry-title" className="text-sm font-medium">Entry conditions (AND)</div>
                <div data-testid="backtest-entry-subtitle" className="text-xs text-muted-foreground">All must be true to enter.</div>
              </div>
              <Button data-testid="backtest-add-entry-condition" type="button" variant="outline" className="rounded-full" onClick={() => addCondition(setEntry)}>
                Add entry
              </Button>
            </div>
            <div className="space-y-3">
              {entry.map((c, idx) => (
                <ConditionEditor
                  key={idx}
                  idx={idx}
                  value={c}
                  testPrefix="backtest-entry"
                  onChange={(next) => setEntry((prev) => prev.map((p, i) => (i === idx ? next : p)))}
                  onRemove={() => setEntry((prev) => prev.filter((_, i) => i !== idx))}
                />
              ))}
              {entry.length === 0 ? (
                <div data-testid="backtest-entry-empty" className="text-sm text-muted-foreground">
                  Add at least one entry condition.
                </div>
              ) : null}
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div>
                <div data-testid="backtest-exit-title" className="text-sm font-medium">Exit conditions (AND)</div>
                <div data-testid="backtest-exit-subtitle" className="text-xs text-muted-foreground">All must be true to exit.</div>
              </div>
              <Button data-testid="backtest-add-exit-condition" type="button" variant="outline" className="rounded-full" onClick={() => addCondition(setExit)}>
                Add exit
              </Button>
            </div>
            <div className="space-y-3">
              {exit.map((c, idx) => (
                <ConditionEditor
                  key={idx}
                  idx={idx}
                  value={c}
                  testPrefix="backtest-exit"
                  onChange={(next) => setExit((prev) => prev.map((p, i) => (i === idx ? next : p)))}
                  onRemove={() => setExit((prev) => prev.filter((_, i) => i !== idx))}
                />
              ))}
              {exit.length === 0 ? (
                <div data-testid="backtest-exit-empty" className="text-sm text-muted-foreground">
                  Add at least one exit condition.
                </div>
              ) : null}
            </div>
          </div>

          <div className="space-y-3">
            <Label data-testid="backtest-risk-label" className="text-sm font-medium">Risk (optional)</Label>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">Stop-loss %</Label>
                <Input
                  type="number"
                  placeholder="e.g. 1"
                  value={risk.stop_loss_pct ?? ""}
                  onChange={(e) => setRisk((r) => ({ ...r, stop_loss_pct: e.target.value ? Number(e.target.value) : undefined }))}
                  className="rounded-xl"
                />
              </div>
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">Take-profit %</Label>
                <Input
                  type="number"
                  placeholder="e.g. 2"
                  value={risk.take_profit_pct ?? ""}
                  onChange={(e) => setRisk((r) => ({ ...r, take_profit_pct: e.target.value ? Number(e.target.value) : undefined }))}
                  className="rounded-xl"
                />
              </div>
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">Max hold (bars)</Label>
                <Input
                  type="number"
                  placeholder="e.g. 60"
                  value={risk.max_hold_bars ?? ""}
                  onChange={(e) => setRisk((r) => ({ ...r, max_hold_bars: e.target.value ? Number(e.target.value) : undefined }))}
                  className="rounded-xl"
                />
              </div>
            </div>
          </div>

          <div data-testid="backtest-mvp-note" className="rounded-2xl border bg-card/60 p-4 text-sm text-muted-foreground">
            Long-only; execute at next bar open. Signal metrics (OBI, aggression, absorption) use backend time-series when available.
          </div>

          {error ? (
            <div data-testid="backtest-error" className="text-sm text-destructive">
              {error}
            </div>
          ) : null}
        </CardContent>
      </Card>

      {run ? (
        <div data-testid="backtest-results" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <Card data-testid="backtest-summary-card" className="rounded-2xl lg:col-span-1">
              <CardHeader>
                <CardTitle data-testid="backtest-summary-title" className="text-base">Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <div data-testid="backtest-summary-totalreturn-label" className="text-sm text-muted-foreground">Total return</div>
                  <div data-testid="backtest-summary-totalreturn" className="text-sm font-medium">{fmt(run.summary.total_return_pct, 2)}%</div>
                </div>
                <div className="flex items-center justify-between">
                  <div data-testid="backtest-summary-maxdd-label" className="text-sm text-muted-foreground">Max drawdown</div>
                  <div data-testid="backtest-summary-maxdd" className="text-sm font-medium">{fmt(run.summary.max_drawdown_pct, 2)}%</div>
                </div>
                <div className="flex items-center justify-between">
                  <div data-testid="backtest-summary-winrate-label" className="text-sm text-muted-foreground">Win rate</div>
                  <div data-testid="backtest-summary-winrate" className="text-sm font-medium">{fmt(run.summary.win_rate_pct, 1)}%</div>
                </div>
                <div className="flex items-center justify-between">
                  <div data-testid="backtest-summary-trades-label" className="text-sm text-muted-foreground">Trades</div>
                  <div data-testid="backtest-summary-trades" className="text-sm font-medium">{run.summary.trades}</div>
                </div>
                <div className="pt-2">
                  <Badge data-testid="backtest-summary-runid" variant="secondary" className="rounded-full">
                    Run {run.id.slice(0, 8)}…
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card data-testid="backtest-equity-card" className="rounded-2xl lg:col-span-2">
              <CardHeader>
                <CardTitle data-testid="backtest-equity-title" className="text-base">Equity curve</CardTitle>
              </CardHeader>
              <CardContent>
                <div data-testid="backtest-equity-chart" className="h-[260px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={equitySeries} margin={{ left: 6, right: 6, top: 10, bottom: 0 }}>
                      <defs>
                        <linearGradient id="tmxEq" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="hsl(var(--chart-4))" stopOpacity={0.35} />
                          <stop offset="95%" stopColor="hsl(var(--chart-4))" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <XAxis hide dataKey="t" />
                      <YAxis hide />
                      <Tooltip
                        contentStyle={{ background: "hsl(var(--popover))", borderRadius: 12, border: "1px solid hsl(var(--border))" }}
                        labelFormatter={() => ""}
                        formatter={(val) => [fmt(val, 2), "equity"]}
                      />
                      <Area type="monotone" dataKey="equity" stroke="hsl(var(--chart-4))" strokeWidth={2} fill="url(#tmxEq)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <div data-testid="backtest-equity-hint" className="mt-2 text-xs text-muted-foreground">
                  Strategy executes at next bar open. Fee/slippage are applied.
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card data-testid="backtest-drawdown-card" className="rounded-2xl">
              <CardHeader>
                <CardTitle data-testid="backtest-drawdown-title" className="text-base">Drawdown</CardTitle>
              </CardHeader>
              <CardContent>
                <div data-testid="backtest-drawdown-chart" className="h-[260px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={ddSeries} margin={{ left: 8, right: 8, top: 10, bottom: 0 }}>
                      <defs>
                        <linearGradient id="tmxDd" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="hsl(var(--destructive))" stopOpacity={0.25} />
                          <stop offset="95%" stopColor="hsl(var(--destructive))" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <XAxis hide dataKey="t" />
                      <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{ background: "hsl(var(--popover))", borderRadius: 12, border: "1px solid hsl(var(--border))" }}
                        labelFormatter={() => ""}
                        formatter={(val) => [fmt(val, 2), "drawdown %"]}
                      />
                      <Area type="monotone" dataKey="drawdown" stroke="hsl(var(--destructive))" strokeWidth={2} fill="url(#tmxDd)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <div data-testid="backtest-drawdown-hint" className="mt-2 text-xs text-muted-foreground">
                  Drawdown is plotted negative (down from peak).
                </div>
              </CardContent>
            </Card>

            <Card data-testid="backtest-price-trades-card" className="rounded-2xl">
              <CardHeader>
                <CardTitle data-testid="backtest-price-trades-title" className="text-base">Price + trade markers</CardTitle>
              </CardHeader>
              <CardContent>
                <div data-testid="backtest-price-trades-chart" className="h-[260px]">
                  {backtestCandles.length ? (
                    <PriceChart
                      candles={backtestCandles}
                      height={260}
                      isVisible={true}
                      intervalSeconds={60}
                      entryExitMarkers={entryExitMarkersForChart}
                    />
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ left: 8, right: 8, top: 10, bottom: 0 }}>
                        <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="4 6" />
                        <XAxis dataKey="t" hide />
                        <YAxis dataKey="price" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                        <Tooltip
                          contentStyle={{ background: "hsl(var(--popover))", borderRadius: 12, border: "1px solid hsl(var(--border))" }}
                          labelFormatter={() => ""}
                          formatter={(val, name) => [fmt(val, name === "price" ? 2 : 0), name]}
                        />
                        <Scatter data={equitySeries.map((p) => ({ t: p.t, price: p.price }))} fill="hsl(var(--chart-2))" />
                        <Scatter data={tradeMarkers.filter((m) => m.kind === "entry")} fill="hsl(var(--chart-5))" />
                        <Scatter data={tradeMarkers.filter((m) => m.kind === "exit")} fill="hsl(var(--chart-4))" />
                      </ScatterChart>
                    </ResponsiveContainer>
                  )}
                </div>
                <div data-testid="backtest-price-trades-hint" className="mt-2 text-xs text-muted-foreground">
                  {backtestCandles.length ? "OHLC with entry (green) / exit (amber) markers. Backtest entry triggers on next candle open." : "Loading candles… Dots show price; markers indicate entry/exit."}
                </div>
              </CardContent>
            </Card>
          </div>

          <Card data-testid="backtest-trades-card" className="rounded-2xl">
            <CardHeader>
              <CardTitle data-testid="backtest-trades-title" className="text-base">Trades</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="rounded-xl border overflow-hidden">
                <Table data-testid="backtest-trades-table">
                  <TableHeader>
                    <TableRow>
                      <TableHead data-testid="backtest-trades-head-entry">Entry</TableHead>
                      <TableHead data-testid="backtest-trades-head-exit">Exit</TableHead>
                      <TableHead data-testid="backtest-trades-head-ret">Return</TableHead>
                      <TableHead data-testid="backtest-trades-head-pnl">PnL</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {run.trades.length ? (
                      run.trades.slice(0, 200).map((t, i) => (
                        <TableRow data-testid={`backtest-trade-row-${i}`} key={i}>
                          <TableCell data-testid={`backtest-trade-entry-${i}`} className="text-xs">
                            {formatTime(t.entry_time, useUTC)} @ {fmt(t.entry_price, 2)}
                          </TableCell>
                          <TableCell data-testid={`backtest-trade-exit-${i}`} className="text-xs">
                            {formatTime(t.exit_time, useUTC)} @ {fmt(t.exit_price, 2)}
                          </TableCell>
                          <TableCell data-testid={`backtest-trade-return-${i}`} className="text-xs">
                            {fmt(t.return_pct, 2)}%
                          </TableCell>
                          <TableCell data-testid={`backtest-trade-pnl-${i}`} className="text-xs">
                            {fmt(t.pnl, 2)}
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell data-testid="backtest-trades-empty" colSpan={4} className="text-sm text-muted-foreground">
                          No trades triggered in the selected window.
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </div>
      ) : (
        <Card data-testid="backtest-results-empty" className="rounded-2xl">
          <CardContent className="py-8 text-sm text-muted-foreground">
            Run a backtest to see results.
          </CardContent>
        </Card>
      )}
    </div>
  );
}
