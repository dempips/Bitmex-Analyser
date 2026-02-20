import React from "react";
import { cn } from "@/lib/utils";

/**
 * 1–3 blinking arrows indicating direction and strength (e.g. aggression velocity).
 * value: -1 to 1 (negative = down, positive = up). Magnitude maps to 1–3 arrows.
 */
export function VelocityArrows({ value, className = "" }) {
  const v = Math.max(-1, Math.min(1, Number(value) || 0));
  const abs = Math.abs(v);
  const count = abs < 1 / 3 ? 1 : abs < 2 / 3 ? 2 : 3;
  const isUp = v > 0;

  return (
    <span
      className={cn("inline-flex items-center gap-0.5 font-bold text-chart-2", !isUp && "text-destructive", className)}
      aria-hidden
    >
      {Array.from({ length: count }, (_, i) => (
        <span
          key={i}
          className="animate-pulse"
          style={{ animationDelay: `${i * 0.15}s`, animationDuration: "1s" }}
        >
          {isUp ? "↑" : "↓"}
        </span>
      ))}
    </span>
  );
}

/**
 * Spectrum gauge for values from -1 to 1 (e.g. OBI, aggressive imbalance).
 * Shows a horizontal bar with left/right labels and a marker at the current value.
 */
export function SpectrumGauge({ value, leftLabel, rightLabel, title, className = "" }) {
  const clamped = Math.max(-1, Math.min(1, Number(value) || 0));
  const percent = 50 + clamped * 50;

  return (
    <div className={cn("space-y-1", className)}>
      {title ? <div className="text-xs font-medium text-muted-foreground">{title}</div> : null}
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-destructive/80 via-muted-foreground/40 to-chart-2"
          style={{ width: "100%" }}
        />
        <div
          className="absolute top-0 bottom-0 w-0.5 rounded-full bg-foreground shadow-sm"
          style={{ left: `${percent}%`, transform: "translateX(-50%)" }}
        />
      </div>
    </div>
  );
}

/**
 * Simple 0–100% gauge (e.g. absorption).
 */
export function PercentGauge({ value, label, title, className = "" }) {
  const pct = Math.max(0, Math.min(100, Number(value) || 0));

  return (
    <div className={cn("space-y-1", className)}>
      {title ? <div className="text-xs font-medium text-muted-foreground">{title}</div> : null}
      <div className="flex justify-between text-xs">
        {label ? <span className="text-muted-foreground">{label}</span> : <span />}
        <span className="font-medium tabular-nums">{Math.round(pct)}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-chart-5 transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
