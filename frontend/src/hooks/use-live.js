import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/api/client";

export function useLive() {
  const [status, setStatus] = useState({ running: false, symbol: "XBTUSD", clients: 0, book_ready: false });
  const [connected, setConnected] = useState(false);
  const [lastMessageAt, setLastMessageAt] = useState(null);

  const [orderbook, setOrderbook] = useState(null);
  const [trades, setTrades] = useState([]);
  const [midSeries, setMidSeries] = useState([]);

  const messageTimesRef = useRef([]);
  const esRef = useRef(null);

  async function refreshStatus() {
    const res = await api.get("/live/status");
    setStatus(res.data);
    return res.data;
  }

  function disconnect() {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    setConnected(false);
  }

  function connect() {
    disconnect();
    const base = process.env.REACT_APP_BACKEND_URL;
    const url = `${base}/api/live/stream`;
    const es = new EventSource(url);
    esRef.current = es;

    es.onopen = () => {
      setConnected(true);
    };

    es.onerror = () => {
      setConnected(false);
      // browser auto-retries; we keep state as disconnected
    };

    es.onmessage = (evt) => {
      const now = Date.now();
      setLastMessageAt(now);
      messageTimesRef.current = [...messageTimesRef.current, now].slice(-200);
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "orderbook") {
          setOrderbook(msg);
          if (msg.mid) {
            setMidSeries((prev) => {
              const next = [...prev, { t: msg.ts, mid: msg.mid }].slice(-180);
              return next;
            });
          }
        } else if (msg.type === "trade") {
          setTrades((prev) => {
            const next = [...prev, msg].slice(-200);
            return next;
          });
        } else if (msg.type === "ws_status") {
          setStatus((s) => ({ ...s, symbol: msg.symbol || s.symbol }));
        }
      } catch (e) {
        // ignore
      }
    };
  }

  async function start(symbol) {
    await api.post("/live/start", { symbol });
    await refreshStatus();
    connect();
  }

  async function restart(symbol) {
    await api.post("/live/restart", { symbol });
    await refreshStatus();
    connect();
  }

  async function stop() {
    await api.post("/live/stop");
    await refreshStatus();
    disconnect();
  }

  useEffect(() => {
    refreshStatus().catch(() => {});
    connect();
    return () => disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const lastTrade = useMemo(() => (trades.length ? trades[trades.length - 1] : null), [trades]);

  const liveMsgsPerSec = useMemo(() => {
    const now = Date.now();
    const recent = messageTimesRef.current.filter((t) => now - t <= 5000);
    return recent.length / 5;
  }, [lastMessageAt]);

  return {
    status,
    connected,
    lastMessageAt,
    liveMsgsPerSec,
    orderbook,
    trades,
    lastTrade,
    midSeries,
    refreshStatus,
    connect,
    disconnect,
    start,
    restart,
    stop,
  };
}
