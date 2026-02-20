import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

import websockets


logger = logging.getLogger(__name__)

# Configurable so you can override (e.g. testnet, or when mainnet DNS fails)
BITMEX_WS_URL = os.environ.get("BITMEX_WS_URL", "wss://www.bitmex.com/realtime")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class BitmexWsConfig:
    symbol: str = "XBTUSD"
    enabled: bool = True


class BitmexWsManager:
    """Single background WS connection that broadcasts messages to all connected clients.

    - Subscriptions: orderBookL2_25 + trade
    - One symbol at a time (server-wide) for MVP simplicity
    - Optional callback on orderbook updates (for persistence/heatmap)
    """

    def __init__(self):
        self.config = BitmexWsConfig()
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._clients: Set[asyncio.Queue] = set()

        # book state: id -> {price, size, side}
        self._book: Dict[int, Dict[str, Any]] = {}
        self._book_ready = False

        self.latest_market: Dict[str, Any] = {"ts": now_iso(), "symbol": self.config.symbol}

        # Optional async callback invoked with latest_market payload
        self.on_orderbook: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "symbol": self.config.symbol,
            "running": self._task is not None and not self._task.done(),
            "clients": len(self._clients),
            "book_ready": self._book_ready,
            "ts": now_iso(),
        }

    async def start(self):
        if self._task and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        if not self._task:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=8)
        except Exception:
            pass
        self._task = None

    async def restart(self, symbol: Optional[str] = None):
        if symbol:
            self.config.symbol = symbol
        # reset book to avoid mixing symbols
        self._book = {}
        self._book_ready = False
        self.latest_market = {"ts": now_iso(), "symbol": self.config.symbol}

        await self.stop()
        await self.start()

    async def subscribe_client(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._clients.add(q)

        await self._safe_put(q, {"type": "hello", "ts": now_iso(), "symbol": self.config.symbol})
        if self.latest_market:
            await self._safe_put(q, {"type": "snapshot", **self.latest_market})
        return q

    def unsubscribe_client(self, q: asyncio.Queue):
        if q in self._clients:
            self._clients.remove(q)

    async def broadcast(self, payload: Dict[str, Any]):
        dead: List[asyncio.Queue] = []
        for q in list(self._clients):
            ok = await self._safe_put(q, payload)
            if not ok:
                dead.append(q)
        for q in dead:
            self.unsubscribe_client(q)

    async def _safe_put(self, q: asyncio.Queue, payload: Dict[str, Any]) -> bool:
        try:
            q.put_nowait(payload)
            return True
        except Exception:
            return False

    def _apply_orderbook_msg(self, msg: Dict[str, Any]):
        action = msg.get("action")
        data = msg.get("data") or []

        if action == "partial":
            self._book = {}
            for row in data:
                if row.get("symbol") != self.config.symbol:
                    continue
                _id = row.get("id")
                if _id is None:
                    continue
                self._book[int(_id)] = {
                    "id": int(_id),
                    "price": float(row.get("price") or 0),
                    "size": float(row.get("size") or 0),
                    "side": row.get("side"),
                }
            self._book_ready = True
            return

        if not self._book_ready:
            return

        if action == "insert":
            for row in data:
                _id = row.get("id")
                if _id is None:
                    continue
                self._book[int(_id)] = {
                    "id": int(_id),
                    "price": float(row.get("price") or 0),
                    "size": float(row.get("size") or 0),
                    "side": row.get("side"),
                }
        elif action == "update":
            for row in data:
                _id = row.get("id")
                if _id is None:
                    continue
                cur = self._book.get(int(_id))
                if not cur:
                    continue
                if "size" in row:
                    cur["size"] = float(row.get("size") or 0)
        elif action == "delete":
            for row in data:
                _id = row.get("id")
                if _id is None:
                    continue
                self._book.pop(int(_id), None)

    def _compute_book_top(self) -> Dict[str, Any]:
        bids = [v for v in self._book.values() if v.get("side") == "Buy" and v.get("price")]
        asks = [v for v in self._book.values() if v.get("side") == "Sell" and v.get("price")]
        if not bids or not asks:
            return {}

        bids.sort(key=lambda r: r["price"], reverse=True)
        asks.sort(key=lambda r: r["price"])

        best_bid = bids[0]["price"]
        best_ask = asks[0]["price"]
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        top_bids = [{"price": r["price"], "size": r["size"]} for r in bids[:25]]
        top_asks = [{"price": r["price"], "size": r["size"]} for r in asks[:25]]

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
            "spread": spread,
            "bids": top_bids,
            "asks": top_asks,
        }

    async def _run_loop(self):
        backoff = 1
        while not self._stop_event.is_set() and self.config.enabled:
            try:
                await self.broadcast({"type": "ws_status", "status": "connecting", "ts": now_iso(), "symbol": self.config.symbol})

                async with websockets.connect(BITMEX_WS_URL, ping_interval=20, ping_timeout=20) as ws:
                    backoff = 1

                    args = [
                        f"orderBookL2_25:{self.config.symbol}",
                        f"trade:{self.config.symbol}",
                    ]
                    await ws.send(json.dumps({"op": "subscribe", "args": args}))

                    await self.broadcast({"type": "ws_status", "status": "connected", "ts": now_iso(), "symbol": self.config.symbol})

                    while not self._stop_event.is_set():
                        raw = await ws.recv()
                        msg = json.loads(raw)

                        table = msg.get("table")
                        if table == "orderBookL2_25":
                            self._apply_orderbook_msg(msg)
                            top = self._compute_book_top()
                            if top:
                                self.latest_market = {"ts": now_iso(), "symbol": self.config.symbol, **top}
                                await self.broadcast({"type": "orderbook", **self.latest_market})
                                if self.on_orderbook:
                                    asyncio.create_task(self.on_orderbook(self.latest_market))

                        elif table == "trade":
                            data = msg.get("data") or []
                            if data:
                                last = data[-1]
                                await self.broadcast(
                                    {
                                        "type": "trade",
                                        "ts": now_iso(),
                                        "symbol": last.get("symbol"),
                                        "side": last.get("side"),
                                        "size": last.get("size"),
                                        "price": last.get("price"),
                                        "trade_ts": last.get("timestamp"),
                                    }
                                )

                        elif "info" in msg:
                            await self.broadcast({"type": "ws_info", "ts": now_iso(), "payload": msg})

            except Exception as e:
                logger.warning("BitMEX WS error: %s", e)
                await self.broadcast({"type": "ws_status", "status": "disconnected", "ts": now_iso(), "symbol": self.config.symbol})
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 20)

        await self.broadcast({"type": "ws_status", "status": "stopped", "ts": now_iso(), "symbol": self.config.symbol})
