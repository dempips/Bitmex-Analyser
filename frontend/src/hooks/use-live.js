import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/api/client";

export function useLive() {
  const [status, setStatus] = useState({ running: false, symbol: "XBTUSD", clients: 0, book_ready: false });
  const [connected, setConnected] = useState(false);
  const [lastMessageAt, setLastMessageAt] = useState(null);

  const [orderbook, setOrderbook] = useState(null);
  const [trades, setTrades] = useState([]);

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
      setLastMessageAt(Date.now());
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "orderbook") {
          setOrderbook(msg);
        } else if (msg.type === "trade") {
          setTrades((prev) => {
            const next = [...prev, msg].slice(-200);
            return next;
          });
        } else if (msg.type === "ws_status") {
          // refresh status lightly
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

  return {
    status,
    connected,
    lastMessageAt,
    orderbook,
    trades,
    lastTrade,
    refreshStatus,
    connect,
    disconnect,
    start,
    restart,
    stop,
  };
}
