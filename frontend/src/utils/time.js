// Time/Timezone utilities shared across pages.

export const TIMEZONE_STORAGE_KEY = "tmx_timezone"; // "utc" | "local"

export function getStoredTimezone(defaultValue = "utc") {
  try {
    const v = localStorage.getItem(TIMEZONE_STORAGE_KEY);
    return v === "local" || v === "utc" ? v : defaultValue;
  } catch {
    return defaultValue;
  }
}

export function setStoredTimezone(v) {
  try {
    localStorage.setItem(TIMEZONE_STORAGE_KEY, v);
  } catch {
    // ignore
  }
}

/** ts can be seconds, ms, Date, or an ISO/date string. */
export function formatTime(ts, useUTC) {
  if (ts == null) return "—";
  const d =
    ts instanceof Date
      ? ts
      : new Date(typeof ts === "number" ? (ts < 1e12 ? ts * 1000 : ts) : ts);
  if (Number.isNaN(d.getTime())) return "—";
  return useUTC
    ? `${d.toISOString().replace("T", " ").slice(0, 19)} UTC`
    : d.toLocaleString();
}

/**
 * Build a `datetime-local` value.
 * - If asUTC=true: "YYYY-MM-DDTHH:mm" representing UTC clock-time.
 * - Else: the user's local clock-time.
 */
export function dateToDateTimeLocalValue(date, asUTC) {
  const d = date instanceof Date ? date : new Date(date);
  if (Number.isNaN(d.getTime())) return "";
  if (asUTC) return d.toISOString().slice(0, 16);
  // Convert to "local ISO" by removing TZ offset before slicing.
  const local = new Date(d.getTime() - d.getTimezoneOffset() * 60_000);
  return local.toISOString().slice(0, 16);
}

/**
 * Parse a `datetime-local` value into a Date.
 * - If asUTC=true, treat the input as UTC ("...Z")
 * - Else, treat as local time.
 */
export function dateTimeLocalValueToDate(value, asUTC) {
  if (!value) return null;
  const d = asUTC ? new Date(`${value}Z`) : new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

/**
 * Convert a `datetime-local` string to preserve the instant when toggling timezone mode.
 */
export function convertDateTimeLocalValue(value, fromUTC, toUTC) {
  const d = dateTimeLocalValueToDate(value, fromUTC);
  if (!d) return value;
  return dateToDateTimeLocalValue(d, toUTC);
}

