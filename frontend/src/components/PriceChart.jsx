import React, { useEffect, useRef, useMemo, useState } from "react";
import { createChart } from "lightweight-charts";

const PRICE_CHART_DATA_MAX = 10000;

/**
 * Resolve a CSS variable. lightweight-charts only parses hex/rgb reliably; we convert to hex.
 */
function cssVar(name, fallback) {
  if (typeof window === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(name.startsWith("--") ? name : `--${name}`)
    .trim();
  if (!value) return fallback;
  // shadcn/tailwind: "h s% l%" -> convert to hex for chart lib
  const parts = value.split(/\s+/);
  if (parts.length >= 3 && value.includes("%")) {
    const h = Number(parts[0]);
    const s = Number(parts[1].replace("%", "")) / 100;
    const l = Number(parts[2].replace("%", "")) / 100;
    return hslToHex(h, s, l);
  }
  if (value.startsWith("#") || value.startsWith("rgb")) return value;
  return fallback;
}

function hslToHex(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const f = (n) => {
    const k = (n + h / 30) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  const r = Math.round(f(0) * 255);
  const g = Math.round(f(8) * 255);
  const b = Math.round(f(4) * 255);
  return `#${[r, g, b].map((x) => x.toString(16).padStart(2, "0")).join("")}`;
}

function timestampToUtc(ts) {
  if (typeof ts === "number") return ts < 1e12 ? ts : Math.floor(ts / 1000);
  const d = new Date(ts);
  return Math.floor(d.getTime() / 1000);
}

function toUtcSeconds(ts) {
  if (ts == null) return null;
  if (typeof ts === "number") return ts < 1e12 ? ts : Math.floor(ts / 1000);
  return Math.floor(new Date(ts).getTime() / 1000);
}

/** Ensure time is a number (Unix seconds) for lightweight-charts; they reject objects. */
function timeToNumber(t) {
  if (t == null) return null;
  if (typeof t === "number" && !Number.isNaN(t)) return t < 1e12 ? t : Math.floor(t / 1000);
  if (typeof t === "string" || typeof t === "object") {
    const n = Math.floor(new Date(t).getTime() / 1000);
    return Number.isNaN(n) ? null : n;
  }
  return null;
}

export function PriceChart({
  candles = [],
  cvdSeries = [],
  flowBars = [],
  entryExitMarkers = [],
  height = 380,
  className = "",
  isVisible = true,
  livePrice = null,
  intervalSeconds = 60,
}) {
  const wrapperRef = useRef(null);
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const chartCreatedRef = useRef(false);
  const formingBarRef = useRef(null);
  const [measureTick, setMeasureTick] = useState(0);
  const [chartReady, setChartReady] = useState(false);

  const ohlcData = useMemo(() => {
    const mapped = candles.map((c) => ({
      time: timestampToUtc(c.timestamp),
      open: Number(c.open),
      high: Number(c.high),
      low: Number(c.low),
      close: Number(c.close),
    }));
    return mapped.length > PRICE_CHART_DATA_MAX ? mapped.slice(-PRICE_CHART_DATA_MAX) : mapped;
  }, [candles]);

  const volumeData = useMemo(() => {
    if (!flowBars.length) return [];
    const mapped = flowBars.map((b) => {
      const t = typeof b.t === "string" ? b.t : b.t;
      const buy = Number(b.buy) || 0;
      const sell = Math.abs(Number(b.sell)) || 0;
      const total = buy + sell;
      const isUp = buy >= sell;
      return {
        time: timestampToUtc(t),
        value: total,
        color: isUp ? "rgba(34, 197, 94, 0.6)" : "rgba(239, 68, 68, 0.6)",
      };
    });
    return mapped.length > PRICE_CHART_DATA_MAX ? mapped.slice(-PRICE_CHART_DATA_MAX) : mapped;
  }, [flowBars]);

  const cvdData = useMemo(() => {
    const mapped = cvdSeries.map((p) => ({
      time: timestampToUtc(p.t),
      value: Number(p.cvd),
    }));
    return mapped.length > PRICE_CHART_DATA_MAX ? mapped.slice(-PRICE_CHART_DATA_MAX) : mapped;
  }, [cvdSeries]);

  const markers = useMemo(() => {
    const sortedCandleTimes = ohlcData.length ? [...new Set(ohlcData.map((d) => d.time))].sort((a, b) => a - b) : [];
    const snapToCandle = (timeSec) => {
      if (!sortedCandleTimes.length) return timeSec;
      let best = sortedCandleTimes[0];
      let bestDiff = Math.abs(timeSec - best);
      for (let i = 0; i < sortedCandleTimes.length; i++) {
        const d = Math.abs(timeSec - sortedCandleTimes[i]);
        if (d < bestDiff) {
          bestDiff = d;
          best = sortedCandleTimes[i];
        }
      }
      return best;
    };
    const fromTrades = (entryExitMarkers || []).map((m) => {
      const timeSec = timestampToUtc(m.t);
      return {
        time: snapToCandle(timeSec),
        position: m.kind === "exit" ? "aboveBar" : "belowBar",
        color: m.kind === "exit" ? "#f59e0b" : "#22c55e",
        shape: m.kind === "exit" ? "arrowDown" : "arrowUp",
      };
    });
    return fromTrades.length > PRICE_CHART_DATA_MAX ? fromTrades.slice(-PRICE_CHART_DATA_MAX) : fromTrades;
  }, [entryExitMarkers, ohlcData]);

  // Create chart ONCE when container is visible and has non-zero size; update via setData/update only after.
  useEffect(() => {
    if (!isVisible || !containerRef.current || !ohlcData.length) return;
    const el = containerRef.current;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    if (w <= 0 || h <= 0) {
      const id = requestAnimationFrame(() => setMeasureTick((t) => t + 1));
      return () => cancelAnimationFrame(id);
    }

    if (chartCreatedRef.current && chartRef.current) {
      chartRef.current.chart.resize(w, h);
      return;
    }

    const border = cssVar("border", "#27272a");
    const text = cssVar("muted-foreground", "#71717a");
    const grid = cssVar("border", "#27272a");
    const popover = cssVar("popover", "#18181b");
    const chart2 = cssVar("chart-2", "#22c55e");
    const downRed = cssVar("destructive", "#ef4444");
    const chart5 = cssVar("chart-5", "#8b5cf6");

    const chart = createChart(el, {
      layout: {
        background: { type: "solid", color: "transparent" },
        textColor: text,
        fontFamily: "inherit",
      },
      grid: {
        vertLines: { color: grid },
        horzLines: { color: grid },
      },
      crosshair: {
        mode: 1,
        vertLine: { labelBackgroundColor: popover },
        horzLine: { labelBackgroundColor: popover },
      },
      rightPriceScale: { borderColor: border, scaleMargins: { top: 0.1, bottom: 0.25 } },
      timeScale: { borderColor: border, timeVisible: true, secondsVisible: false },
      width: w,
      height: h,
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: chart2,
      downColor: downRed,
      borderVisible: true,
      wickUpColor: chart2,
      wickDownColor: downRed,
    });
    candlestickSeries.setData(ohlcData);
    if (markers.length) candlestickSeries.setMarkers(markers);

    let volumeSeries = null;
    if (volumeData.length) {
      volumeSeries = chart.addHistogramSeries({
        color: "rgba(34, 197, 94, 0.5)",
        priceFormat: { type: "volume" },
        priceScaleId: "",
      });
      volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
      volumeSeries.setData(volumeData);
    }

    let cvdLineSeries = null;
    if (cvdData.length) {
      cvdLineSeries = chart.addLineSeries({
        color: chart5,
        lineWidth: 2,
        priceScaleId: "cvd",
      });
      chart.priceScale("cvd").applyOptions({ scaleMargins: { top: 0.85, bottom: 0 }, borderVisible: false });
      cvdLineSeries.setData(cvdData);
    }

    chart.timeScale().fitContent();
    chartRef.current = { chart, candlestickSeries, volumeSeries, cvdLineSeries };
    chartCreatedRef.current = true;
    setChartReady(true);
  }, [isVisible, height, ohlcData.length, measureTick]);

  // Resize when tab becomes visible again (container gets dimensions back).
  useEffect(() => {
    if (!isVisible || !chartRef.current || !containerRef.current) return;
    const el = containerRef.current;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    if (w > 0 && h > 0) chartRef.current.chart.resize(w, h);
  }, [isVisible]);

  // ResizeObserver: when Price tab is shown, container may get dimensions after layout; ensure chart inits or resizes.
  useEffect(() => {
    if (!isVisible || !containerRef.current) return;
    const el = containerRef.current;
    const ro = new ResizeObserver(() => {
      if (!chartRef.current?.chart) setMeasureTick((t) => t + 1);
      else if (el.offsetWidth > 0 && el.offsetHeight > 0) chartRef.current.chart.resize(el.offsetWidth, el.offsetHeight);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [isVisible]);

  // Destroy chart only on unmount; clear create guard so remount can create again.
  useEffect(() => {
    return () => {
      if (chartRef.current?.chart) {
        chartRef.current.chart.remove();
        chartRef.current = null;
      }
      chartCreatedRef.current = false;
    };
  }, []);

  // Track when base candles actually changed (new symbol/range/timeframe) so we don't reset zoom or wipe live bar.
  const ohlcSignatureRef = useRef("");
  const ohlcSignature = ohlcData.length ? `${ohlcData.length}-${ohlcData[ohlcData.length - 1]?.time}` : "";

  useEffect(() => {
    if (!chartRef.current || !ohlcData.length) return;
    const { chart, candlestickSeries, volumeSeries, cvdLineSeries } = chartRef.current;
    const chart5 = cssVar("chart-5", "#8b5cf6");

    const candlesChanged = ohlcSignatureRef.current !== ohlcSignature;
    if (candlesChanged) {
      ohlcSignatureRef.current = ohlcSignature;
      candlestickSeries.setData(ohlcData);
      candlestickSeries.setMarkers(markers);
      formingBarRef.current = null;
      chart.timeScale().fitContent();
    } else {
      candlestickSeries.setMarkers(markers);
    }

    // Add volume series if we have data but it wasn't created at init (e.g. flow loaded after candles).
    let volSeries = volumeSeries;
    if (volumeData.length && !volSeries) {
      volSeries = chart.addHistogramSeries({
        color: "rgba(34, 197, 94, 0.5)",
        priceFormat: { type: "volume" },
        priceScaleId: "",
      });
      volSeries.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 }, borderVisible: false });
      chartRef.current.volumeSeries = volSeries;
    }
    if (volSeries) volSeries.setData(volumeData.length ? volumeData : []);

    // Add CVD line series if we have data but it wasn't created at init.
    let cvdSeries = cvdLineSeries;
    if (cvdData.length && !cvdSeries) {
      cvdSeries = chart.addLineSeries({
        color: chart5,
        lineWidth: 2,
        priceScaleId: "cvd",
      });
      chart.priceScale("cvd").applyOptions({ scaleMargins: { top: 0.85, bottom: 0 }, borderVisible: false });
      chartRef.current.cvdLineSeries = cvdSeries;
    }
    if (cvdSeries) cvdSeries.setData(cvdData.length ? cvdData : []);
  }, [ohlcData, ohlcSignature, volumeData, cvdData, markers]);

  // Live tick: update forming candle in place or append new bucket; no REST.
  // lightweight-charts: update() only allows updating the LAST bar (same time) or appending (new time > last). Never pass an older time.
  useEffect(() => {
    if (!livePrice || !chartRef.current?.candlestickSeries || !ohlcData.length) return;
    const price = Number(livePrice.price);
    if (Number.isNaN(price)) return;
    const tsSec = toUtcSeconds(livePrice.timestamp);
    if (tsSec == null) return;
    const bucketStartSec = Math.floor(tsSec / intervalSeconds) * intervalSeconds;
    const candlestickSeries = chartRef.current.candlestickSeries;

    let bar = formingBarRef.current;
    const last = ohlcData[ohlcData.length - 1];
    const lastTimeNum = timeToNumber(last?.time ?? null);

    if (bar) {
      const barTimeNum = timeToNumber(bar.time);
      if (barTimeNum === bucketStartSec) {
        bar = {
          time: bucketStartSec,
          open: bar.open,
          high: Math.max(bar.high, price),
          low: Math.min(bar.low, price),
          close: price,
        };
      } else {
        bar = {
          time: bucketStartSec,
          open: bar.close,
          high: price,
          low: price,
          close: price,
        };
      }
    } else {
      const lastTimeFromData = timeToNumber(last?.time);
      if (lastTimeFromData === bucketStartSec) {
        bar = {
          time: bucketStartSec,
          open: Number(last.open),
          high: Math.max(Number(last.high), price),
          low: Math.min(Number(last.low), price),
          close: price,
        };
      } else {
        bar = {
          time: bucketStartSec,
          open: last ? Number(last.close) : price,
          high: price,
          low: price,
          close: price,
        };
      }
    }

    const newTimeNum = timeToNumber(bar.time);
    if (newTimeNum == null) return;
    const lastBarTime = timeToNumber(formingBarRef.current?.time) ?? lastTimeNum;
    if (lastBarTime != null && newTimeNum < lastBarTime) return;

    formingBarRef.current = bar;
    candlestickSeries.update(bar);
  }, [livePrice, ohlcData, intervalSeconds]);

  if (!candles.length) {
    return <div style={{ height }} />;
  }

  return (
    <div
      ref={wrapperRef}
      className={className}
      style={{ position: "relative", width: "100%", height: `${height}px` }}
    >
      <div
        ref={containerRef}
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          width: "100%",
          height: "100%",
          zIndex: 1,
        }}
      />
    </div>
  );
}
