import { useEffect, useMemo, useRef, useState } from "react";
import { api, BACKEND_URL } from "@/api/client";

export function useLive() {
  const [status, setStatus] = useState({ running: false, symbol: "XBTUSD", clients: 0, book_ready: false });
  const [connected, setConnected] = useState(false);
  const [lastMessageAt, setLastMessageAt] = useState(null);

  const [orderbook, setOrderbook] = useState(null);
  const [trades, setTrades] = useState([]);
  const [midSeries, setMidSeries] = useState([]);

  const messageTimesRef = useRef([]);
  const esRef = useRef(null);
  const connectingRef = useRef(false);
  const mountedRef = useRef(true);
  const lastMessageAtRef = useRef(null);
  const openedAtRef = useRef(null);
  const connectedRef = useRef(false);
  const STALE_CONNECTION_MS = 15000; // if no message in 15s, consider dead
  // Cap live buffers in refs to avoid unbounded growth and tab crash; promote to state every 250–500ms.
  const tradesRef = useRef([]);
  const orderbookRef = useRef(null);
  const lastPromoteRef = useRef(0);
  const LIVETRADES_CAP = 500;
  const PROMOTE_INTERVAL_MS = 350;

  function maybePromoteState(now) {
    if (now - lastPromoteRef.current < PROMOTE_INTERVAL_MS) return;
    lastPromoteRef.current = now;
    if (!mountedRef.current) return;
    if (orderbookRef.current) setOrderbook(orderbookRef.current);
    setTrades([...tradesRef.current]);
  }

  async function refreshStatus() {
    try {
      const res = await api.get("/live/status");
      setStatus(res.data);
      return res.data;
    } catch (e) {
      // Don't crash render if backend is temporarily unavailable.
      console.error("live: refreshStatus failed", e);
      return null;
    }
  }

  function disconnect() {
    const sock = esRef.current;
    if (sock) {
      try {
        if (typeof sock.send === "function") {
          sock.onmessage = null;
          sock.onerror = null;
          sock.onclose = null;
          sock.onopen = null;
        }
        if (typeof sock.close === "function") sock.close();
      } catch (e) {
        // ignore
      }
      esRef.current = null;
    }
    connectingRef.current = false;
    tradesRef.current = [];
    orderbookRef.current = null;
    openedAtRef.current = null;
    connectedRef.current = false;
    if (mountedRef.current) {
      setConnected(false);
      setOrderbook(null);
      setTrades([]);
    }
  }

  function connect(force = false) {
    // Guard against duplicate sockets (React 18 StrictMode, user double-clicks, etc).
    if (!force) {
      if (connectingRef.current) return;
      const cur = esRef.current;
      if (cur) {
        // WebSocket.readyState: 0 CONNECTING, 1 OPEN
        if (typeof cur.readyState === "number" && (cur.readyState === 0 || cur.readyState === 1)) return;
        // EventSource.readyState: 0 CONNECTING, 1 OPEN
        if (typeof cur.readyState === "number" && (cur.readyState === 0 || cur.readyState === 1)) return;
      }
    }

    disconnect();
    connectingRef.current = true;

    const base = BACKEND_URL;
    const wsUrl = base.replace(/^http/, "ws") + "/api/live/ws";

    // Prefer native WebSocket, but fall back to SSE if it fails quickly.
    let didOpen = false;
    let fallbackTimer = null;

    try {
      const ws = new WebSocket(wsUrl);
      esRef.current = ws;

      fallbackTimer = setTimeout(() => {
        if (!didOpen) {
          try {
            ws.close();
          } catch (e) {
            // ignore
          }
          // fallback to SSE
          connectSse();
        }
      }, 1500);

      ws.onopen = () => {
        didOpen = true;
        if (fallbackTimer) clearTimeout(fallbackTimer);
        connectingRef.current = false;
        openedAtRef.current = Date.now();
        connectedRef.current = true;
        if (mountedRef.current) setConnected(true);
      };

      ws.onerror = (e) => {
        console.error("live: ws error", e);
        connectingRef.current = false;
        connectedRef.current = false;
        if (mountedRef.current) setConnected(false);
      };

      ws.onclose = () => {
        if (fallbackTimer) clearTimeout(fallbackTimer);
        if (!didOpen) connectSse();
        connectingRef.current = false;
        connectedRef.current = false;
        openedAtRef.current = null;
        if (mountedRef.current) setConnected(false);
      };

      ws.onmessage = (evt) => {
        if (!mountedRef.current) return;
        const now = Date.now();
        lastMessageAtRef.current = now;
        setLastMessageAt(now);
        messageTimesRef.current = [...messageTimesRef.current, now].slice(-200);
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === "orderbook") {
            const depth = 50;
            orderbookRef.current = {
              ...msg,
              bids: (msg.bids || []).slice(0, depth),
              asks: (msg.asks || []).slice(0, depth),
            };
            if (msg.mid) {
              setMidSeries((prev) => {
                const next = [...prev, { t: msg.ts, mid: msg.mid }].slice(-180);
                return next;
              });
            }
            maybePromoteState(now);
          } else if (msg.type === "trade") {
            const prev = tradesRef.current;
            tradesRef.current = [...prev, msg].slice(-LIVETRADES_CAP);
            maybePromoteState(now);
          } else if (msg.type === "ws_status") {
            setStatus((s) => ({ ...s, symbol: msg.symbol || s.symbol }));
          }
        } catch (e) {
          // ignore
        }
      };

      return;
    } catch (e) {
      console.error("live: ws connect failed", e);
      // fallback to SSE
    }

    connectSse();

    function connectSse() {
      const sseUrl = `${BACKEND_URL}/api/live/stream`;
      try {
        const es = new EventSource(sseUrl);
        esRef.current = es;

        es.onopen = () => {
          connectingRef.current = false;
          openedAtRef.current = Date.now();
          connectedRef.current = true;
          if (mountedRef.current) setConnected(true);
        };

        es.onerror = (e) => {
          console.error("live: sse error", e);
          connectingRef.current = false;
          connectedRef.current = false;
          if (mountedRef.current) setConnected(false);
        };

        es.onmessage = (evt) => {
          if (!mountedRef.current) return;
          const now = Date.now();
          lastMessageAtRef.current = now;
          setLastMessageAt(now);
          messageTimesRef.current = [...messageTimesRef.current, now].slice(-200);
          try {
            const msg = JSON.parse(evt.data);
            if (msg.type === "orderbook") {
              const depth = 50;
              orderbookRef.current = {
                ...msg,
                bids: (msg.bids || []).slice(0, depth),
                asks: (msg.asks || []).slice(0, depth),
              };
              if (msg.mid) {
                setMidSeries((prev) => {
                  const next = [...prev, { t: msg.ts, mid: msg.mid }].slice(-180);
                  return next;
                });
              }
              maybePromoteState(now);
            } else if (msg.type === "trade") {
              const prev = tradesRef.current;
              tradesRef.current = [...prev, msg].slice(-LIVETRADES_CAP);
              maybePromoteState(now);
            } else if (msg.type === "ws_status") {
              setStatus((s) => ({ ...s, symbol: msg.symbol || s.symbol }));
            }
          } catch (e) {
            // ignore
          }
        };
      } catch (e) {
        connectingRef.current = false;
        console.error("live: sse connect failed", e);
        setConnected(false);
      }
    }
  }

  async function start(symbol) {
    try {
      await api.post("/live/start", { symbol });
      await refreshStatus();
      connect(true);
    } catch (e) {
      console.error("live: start failed", e);
    }
  }

  async function restart(symbol) {
    try {
      await api.post("/live/restart", { symbol });
      await refreshStatus();
      // Ensure any existing socket is closed before reconnecting.
      connect(true);
    } catch (e) {
      console.error("live: restart failed", e);
    }
  }

  async function stop() {
    try {
      await api.post("/live/stop");
      await refreshStatus();
      disconnect();
    } catch (e) {
      console.error("live: stop failed", e);
    }
  }

  useEffect(() => {
    connectedRef.current = connected;
  }, [connected]);

  useEffect(() => {
    mountedRef.current = true;
    refreshStatus().catch(() => {});

    const handleOffline = () => {
      if (mountedRef.current) {
        connectedRef.current = false;
        disconnect();
      }
    };

    const handleOnline = () => {
      // Reconnection is handled by Analytics.jsx
    };

    window.addEventListener("offline", handleOffline);
    window.addEventListener("online", handleOnline);

    const staleCheckInterval = setInterval(() => {
      if (!connectedRef.current || !esRef.current) return;
      const now = Date.now();
      const openedAt = openedAtRef.current;
      const lastMsg = lastMessageAtRef.current;
      const gracePeriodOver = openedAt != null && now - openedAt > STALE_CONNECTION_MS;
      const noRecentMessage = lastMsg == null || now - lastMsg > STALE_CONNECTION_MS;
      if (gracePeriodOver && noRecentMessage) {
        console.warn("[Live] No message in", STALE_CONNECTION_MS / 1000, "s — marking disconnected");
        disconnect();
      }
    }, 5000);

    return () => {
      mountedRef.current = false;
      window.removeEventListener("offline", handleOffline);
      window.removeEventListener("online", handleOnline);
      clearInterval(staleCheckInterval);
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const lastTrade = useMemo(() => (trades.length ? trades[trades.length - 1] : null), [trades]);

  const liveMsgsPerSec = useMemo(() => {
    const now = Date.now();
    const recent = messageTimesRef.current.filter((t) => now - t <= 5000);
    return recent.length / 5;
    // We intentionally depend on lastMessageAt to recompute periodically.
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
