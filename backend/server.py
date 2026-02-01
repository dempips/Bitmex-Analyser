import os
import uuid
import logging
import asyncio

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Tuple

import jwt
import requests
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status, WebSocket
from starlette.websockets import WebSocketDisconnect

from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from pydantic import BaseModel, Field, ConfigDict
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from ws_manager import BitmexWsManager


# -----------------------------
# Env + DB
# -----------------------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
JWT_EXPIRES_HOURS = int(os.environ.get("JWT_EXPIRES_HOURS", "168"))  # 7 days

BITMEX_BASE_URL = "https://www.bitmex.com/api/v1"


# -----------------------------
# App
# -----------------------------
app = FastAPI()
api_router = APIRouter(prefix="/api")

ws_manager = BitmexWsManager()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------
# Utilities
# -----------------------------
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def floor_to_minute(dt: datetime) -> datetime:
    dt2 = dt.astimezone(timezone.utc)
    return dt2.replace(second=0, microsecond=0)


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


security = HTTPBearer(auto_error=False)


def create_access_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(now_utc().timestamp()),
        "exp": int((now_utc() + timedelta(hours=JWT_EXPIRES_HOURS)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    if not creds or not creds.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    token = creds.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return user
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired") from e
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from e


def mongo_to_user_out(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "created_at": user.get("created_at"),
    }


def op_to_fn(op: str):
    # for numeric comparisons
    if op == ">":
        return lambda a, b: a > b
    if op == ">=":
        return lambda a, b: a >= b
    if op == "<":
        return lambda a, b: a < b
    if op == "<=":
        return lambda a, b: a <= b
    if op == "==":
        return lambda a, b: a == b
    raise ValueError("Unsupported operator")


# -----------------------------
# Pydantic Models
# -----------------------------
class HealthResponse(BaseModel):
    ok: bool


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: Dict[str, Any]


class MeResponse(BaseModel):
    user: Dict[str, Any]


class SymbolItem(BaseModel):
    symbol: str
    typ: Optional[str] = None
    state: Optional[str] = None
    listing: Optional[str] = None


class Candle(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class BandDepth(BaseModel):
    band_bps: int
    bid_depth: float
    ask_depth: float
    obi: float
    weighted_obi: float


class AnalyticsSnapshotResponse(BaseModel):
    symbol: str
    ts: str
    best_bid: float
    best_ask: float
    mid: float
    spread: float
    bands: List[BandDepth]


class FlowResponse(BaseModel):
    symbol: str
    ts: str
    minutes: int
    buy_volume: float
    sell_volume: float
    aggressive_imbalance: float
    cvd: float
    price_change: float
    absorption_ratio: float


class OrderBookLevel(BaseModel):
    price: float
    size: float


class OrderBookL2Response(BaseModel):
    symbol: str
    ts: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]


class DepthPoint(BaseModel):
    price: float
    cum_size: float


class OrderBookDepthResponse(BaseModel):
    symbol: str
    ts: str
    bids: List[DepthPoint]
    asks: List[DepthPoint]


class FlowSeriesPoint(BaseModel):
    t: str
    buy: float
    sell: float
    delta: float
    cvd: float
    close: Optional[float] = None


class FlowSeriesResponse(BaseModel):
    symbol: str
    ts: str
    minutes: int
    points: List[FlowSeriesPoint]


class FundingPoint(BaseModel):
    t: str
    funding_rate: float
    momentum: float


class FundingResponse(BaseModel):
    symbol: str
    ts: str
    points: List[FundingPoint]


class OpenInterestPoint(BaseModel):
    t: str
    open_interest: float
    delta: float


class OpenInterestResponse(BaseModel):
    symbol: str
    ts: str
    points: List[OpenInterestPoint]


class LiquidationPoint(BaseModel):
    t: str
    price: float
    size: float
    side: Optional[str] = None


class LiquidationsResponse(BaseModel):
    symbol: str
    ts: str
    minutes: int
    points: List[LiquidationPoint]


class HeatmapCell(BaseModel):
    t: str
    side: Literal["Buy", "Sell"]
    price: float
    size: float


class HeatmapResponse(BaseModel):
    symbol: str
    ts: str
    minutes: int
    cells: List[HeatmapCell]


ConditionMetric = Literal["close", "return_1", "sma", "ema", "volatility"]


class StrategyCondition(BaseModel):
    metric: ConditionMetric
    period: Optional[int] = None  # for sma/ema/volatility
    operator: Literal[">", ">=", "<", "<=", "=="]
    value: float


class StrategyCreate(BaseModel):
    name: str
    symbol: str
    entry_conditions: List[StrategyCondition]
    exit_conditions: List[StrategyCondition]
    fee_bps: float = 7.5
    slippage_bps: float = 2.0


class StrategyOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    user_id: str
    name: str
    symbol: str
    entry_conditions: List[StrategyCondition]
    exit_conditions: List[StrategyCondition]
    fee_bps: float
    slippage_bps: float
    created_at: str


class BacktestRunRequest(BaseModel):
    symbol: str
    start: str  # ISO
    end: str  # ISO
    strategy: StrategyCreate
    initial_capital: float = 10000.0


class BacktestTrade(BaseModel):
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    pnl: float
    return_pct: float


class BacktestSummary(BaseModel):
    total_return_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    trades: int


class BacktestRunResponse(BaseModel):
    id: str
    created_at: str
    symbol: str
    start: str
    end: str
    summary: BacktestSummary
    equity_curve: List[Dict[str, Any]]
    trades: List[BacktestTrade]


# -----------------------------
# BitMEX REST helpers
# -----------------------------

def bitmex_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{BITMEX_BASE_URL}{path}"
    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code >= 400:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"BitMEX error: {resp.text[:200]}"
            )
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail="Failed to reach BitMEX") from e


def parse_iso(s: str) -> datetime:
    # accepts Z or +00:00
    s2 = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s2)
    # Ensure timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# -----------------------------
# Indicators (candle-based)
# -----------------------------

def sma(values: List[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if period <= 0:
        return out
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(values: List[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if period <= 0 or not values:
        return out
    k = 2 / (period + 1)
    ema_val: Optional[float] = None
    for i, v in enumerate(values):
        if ema_val is None:
            ema_val = v
        else:
            ema_val = v * k + ema_val * (1 - k)
        out[i] = ema_val if i >= period - 1 else None
    return out


def returns_1(values: List[float]) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    for i in range(1, len(values)):
        prev = values[i - 1]
        out[i] = (values[i] / prev - 1.0) if prev else None
    return out


def rolling_std(values: List[Optional[float]], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if period <= 1:
        return out
    window: List[float] = []
    for i, v in enumerate(values):
        if v is None:
            window.append(float("nan"))
        else:
            window.append(float(v))
        if len(window) > period:
            window.pop(0)
        if len(window) == period and all(not (x != x) for x in window):  # no NaNs
            mean = sum(window) / period
            var = sum((x - mean) ** 2 for x in window) / period
            out[i] = var ** 0.5
    return out


def compute_series(candles: List[Candle]) -> Dict[str, Any]:
    closes = [c.close for c in candles]
    ret1 = returns_1(closes)
    sma10 = sma(closes, 10)
    sma20 = sma(closes, 20)
    sma50 = sma(closes, 50)
    ema10 = ema(closes, 10)
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    vol10 = rolling_std(ret1, 10)
    vol20 = rolling_std(ret1, 20)
    vol50 = rolling_std(ret1, 50)
    return {
        "close": closes,
        "return_1": ret1,
        "sma": {10: sma10, 20: sma20, 50: sma50},
        "ema": {10: ema10, 20: ema20, 50: ema50},
        "volatility": {10: vol10, 20: vol20, 50: vol50},
    }


def eval_condition(series: Dict[str, Any], idx: int, cond: StrategyCondition) -> bool:
    op = op_to_fn(cond.operator)
    if cond.metric == "close":
        v = series["close"][idx]
    elif cond.metric == "return_1":
        v = series["return_1"][idx]
    elif cond.metric in ("sma", "ema", "volatility"):
        if not cond.period or cond.period not in (10, 20, 50):
            return False
        v = series[cond.metric][cond.period][idx]
    else:
        return False

    if v is None:
        return False
    return op(float(v), float(cond.value))


# -----------------------------
# Routes
# -----------------------------
@api_router.get("/", tags=["system"])
async def root():
    return {"message": "TradeMetryx API"}


@api_router.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    return HealthResponse(ok=True)


# -------- WebSocket / Live (SSE) --------
@api_router.get("/live/status", tags=["live"])
async def live_status():
    return ws_manager.status()


class LiveStartRequest(BaseModel):
    symbol: str = "XBTUSD"


@api_router.post("/live/start", tags=["live"])
async def live_start(payload: LiveStartRequest):
    await ws_manager.restart(symbol=payload.symbol)
    return ws_manager.status()


@api_router.post("/live/stop", tags=["live"])
async def live_stop():
    await ws_manager.stop()
    return ws_manager.status()


@api_router.post("/live/restart", tags=["live"])
async def live_restart(payload: LiveStartRequest):
    await ws_manager.restart(symbol=payload.symbol)
    return ws_manager.status()


@api_router.get("/live/stream", tags=["live"])
async def live_stream():
    # Server-Sent Events endpoint.
    # Note: Some proxies buffer streaming responses. We:
    # - disable buffering via headers
    # - send periodic keepalive pings
    q = await ws_manager.subscribe_client()

    async def event_gen():
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=10)
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    # SSE comment ping (keeps connection alive + prevents buffering)
                    yield ": ping\n\n"
        finally:
            ws_manager.unsubscribe_client(q)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)


@api_router.websocket("/live/ws")
async def live_ws(websocket: WebSocket):
    # Native WebSocket stream (preferred if the environment supports it).
    await websocket.accept()
    q = await ws_manager.subscribe_client()
    try:
        while True:
            msg = await q.get()
            await websocket.send_text(json.dumps(msg))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        ws_manager.unsubscribe_client(q)


@api_router.get("/live/heatmap", response_model=HeatmapResponse, tags=["live"])
async def live_heatmap(symbol: str, minutes: int = 10):
    minutes = max(1, min(minutes, 10))
    cutoff = iso(now_utc() - timedelta(minutes=minutes))

    cur = db.orderbook_heat.find({"symbol": symbol, "ts": {"$gte": cutoff}}, {"_id": 0}).sort("ts", 1)
    docs = await cur.to_list(300)

    cells: List[HeatmapCell] = []
    for d in docs:
        t = d.get("ts")
        for b in (d.get("bids") or []):
            if b.get("price") is None:
                continue
            cells.append(HeatmapCell(t=t, side="Buy", price=float(b.get("price") or 0), size=float(b.get("size") or 0)))
        for a in (d.get("asks") or []):
            if a.get("price") is None:
                continue
            cells.append(HeatmapCell(t=t, side="Sell", price=float(a.get("price") or 0), size=float(a.get("size") or 0)))

    return HeatmapResponse(symbol=symbol, ts=iso(now_utc()), minutes=minutes, cells=cells)


# -------- Auth --------
@api_router.post("/auth/register", response_model=AuthResponse, tags=["auth"])
async def register(payload: RegisterRequest):
    email = payload.email.strip().lower()
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    existing = await db.users.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user_doc = {
        "email": email,
        "password_hash": hash_password(payload.password),
        "created_at": iso(now_utc()),
    }
    res = await db.users.insert_one(user_doc)
    user_id = str(res.inserted_id)
    token = create_access_token(user_id=user_id, email=email)
    return AuthResponse(token=token, user=mongo_to_user_out({"_id": res.inserted_id, **user_doc}))


@api_router.post("/auth/login", response_model=AuthResponse, tags=["auth"])
async def login(payload: LoginRequest):
    email = payload.email.strip().lower()
    user = await db.users.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user_id=str(user["_id"]), email=email)
    return AuthResponse(token=token, user=mongo_to_user_out(user))


@api_router.post("/auth/guest", response_model=AuthResponse, tags=["auth"])
async def guest_login():
    # Simple shared guest account for preview/demo.
    # Anyone can use this endpoint; avoid storing any sensitive data under this account.
    guest_email = "guest@trademetryx.local"

    user = await db.users.find_one({"email": guest_email})
    if not user:
        user_doc = {
            "email": guest_email,
            "password_hash": hash_password(str(uuid.uuid4())),
            "created_at": iso(now_utc()),
            "is_guest": True,
        }
        res = await db.users.insert_one(user_doc)
        user = {"_id": res.inserted_id, **user_doc}

    token = create_access_token(user_id=str(user["_id"]), email=guest_email)
    return AuthResponse(token=token, user=mongo_to_user_out(user))


@api_router.get("/auth/me", response_model=MeResponse, tags=["auth"])
async def me(user: Dict[str, Any] = Depends(get_current_user)):
    return MeResponse(user=mongo_to_user_out(user))


# -------- BitMEX data --------
@api_router.get("/bitmex/symbols", response_model=List[SymbolItem], tags=["bitmex"])
async def bitmex_symbols():
    data = bitmex_get("/instrument/active", params=None)
    out: List[SymbolItem] = []
    for row in data:
        sym = row.get("symbol")
        if not sym:
            continue
        out.append(
            SymbolItem(
                symbol=sym,
                typ=row.get("typ"),
                state=row.get("state"),
                listing=row.get("listing"),
            )
        )
    out.sort(key=lambda x: x.symbol)
    return out


@api_router.get("/bitmex/candles", response_model=List[Candle], tags=["bitmex"])
async def bitmex_candles(symbol: str, start: str, end: str):
    start_dt = parse_iso(start)
    end_dt = parse_iso(end)
    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end must be after start")

    all_rows: List[Dict[str, Any]] = []
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(minutes=2000), end_dt)
        params = {
            "symbol": symbol,
            "binSize": "1m",
            "partial": "false",
            "reverse": "false",
            "startTime": iso(cursor),
            "endTime": iso(chunk_end),
            "count": 2000,
        }
        rows = bitmex_get("/trade/bucketed", params=params)
        if not isinstance(rows, list):
            break
        all_rows.extend(rows)

        if rows:
            last_ts = parse_iso(rows[-1]["timestamp"]) + timedelta(minutes=1)
            cursor = max(last_ts, chunk_end)
        else:
            cursor = chunk_end

        if len(all_rows) > 20000:
            raise HTTPException(status_code=400, detail="Requested range too large for MVP")

    candles: List[Candle] = []
    for r in all_rows:
        if not r.get("timestamp"):
            continue
        candles.append(
            Candle(
                timestamp=r["timestamp"],
                open=float(r.get("open") or 0),
                high=float(r.get("high") or 0),
                low=float(r.get("low") or 0),
                close=float(r.get("close") or 0),
                volume=float(r.get("volume") or 0),
            )
        )
    return candles


def compute_orderbook_metrics(
    l2: List[Dict[str, Any]],
    bands_bps: List[int],
) -> Dict[str, Any]:
    bids = [r for r in l2 if r.get("side") == "Buy" and r.get("price") is not None]
    asks = [r for r in l2 if r.get("side") == "Sell" and r.get("price") is not None]
    if not bids or not asks:
        raise HTTPException(status_code=502, detail="BitMEX order book missing sides")

    best_bid = max(float(r["price"]) for r in bids)
    best_ask = min(float(r["price"]) for r in asks)
    mid = (best_bid + best_ask) / 2
    spread = best_ask - best_bid

    band_out: List[BandDepth] = []
    for bps in bands_bps:
        band = mid * (bps / 10000)
        bid_depth = 0.0
        ask_depth = 0.0
        w_bid = 0.0
        w_ask = 0.0

        for r in bids:
            p = float(r["price"])
            if p >= mid - band:
                sz = float(r.get("size") or 0)
                bid_depth += sz
                dist = max(mid - p, 0.0)
                w = 1.0 / (1.0 + (dist / (band + 1e-9)))
                w_bid += sz * w
        for r in asks:
            p = float(r["price"])
            if p <= mid + band:
                sz = float(r.get("size") or 0)
                ask_depth += sz
                dist = max(p - mid, 0.0)
                w = 1.0 / (1.0 + (dist / (band + 1e-9)))
                w_ask += sz * w

        denom = (bid_depth + ask_depth) or 1.0
        obi = (bid_depth - ask_depth) / denom

        w_denom = (w_bid + w_ask) or 1.0
        weighted_obi = (w_bid - w_ask) / w_denom

        band_out.append(
            BandDepth(
                band_bps=bps,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                obi=obi,
                weighted_obi=weighted_obi,
            )
        )

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
        "bands": band_out,
    }


def build_l2_side(levels: List[Dict[str, Any]], side: str) -> List[OrderBookLevel]:
    out: List[OrderBookLevel] = []
    for r in levels:
        if r.get("side") != side:
            continue
        p = r.get("price")
        if p is None:
            continue
        out.append(OrderBookLevel(price=float(p), size=float(r.get("size") or 0)))
    # aggregate by price
    agg: Dict[float, float] = {}
    for lv in out:
        agg[lv.price] = agg.get(lv.price, 0.0) + lv.size
    prices = sorted(agg.keys(), reverse=(side == "Buy"))
    return [OrderBookLevel(price=float(p), size=float(agg[p])) for p in prices]


def cumulative_depth(levels: List[OrderBookLevel]) -> List[DepthPoint]:
    cum = 0.0
    pts: List[DepthPoint] = []
    for lv in levels:
        cum += lv.size
        pts.append(DepthPoint(price=lv.price, cum_size=cum))
    return pts


@api_router.get("/bitmex/orderbook/l2", response_model=OrderBookL2Response, tags=["bitmex"])
async def bitmex_orderbook_l2(symbol: str, depth: int = 50):
    depth = max(10, min(depth, 200))
    l2 = bitmex_get("/orderBook/L2", params={"symbol": symbol, "depth": depth})
    bids = build_l2_side(l2, "Buy")
    asks = build_l2_side(l2, "Sell")
    return OrderBookL2Response(symbol=symbol, ts=iso(now_utc()), bids=bids, asks=asks)


@api_router.get("/bitmex/orderbook/depth", response_model=OrderBookDepthResponse, tags=["bitmex"])
async def bitmex_orderbook_depth(symbol: str, depth: int = 50):
    ob = await bitmex_orderbook_l2(symbol=symbol, depth=depth)
    return OrderBookDepthResponse(
        symbol=symbol,
        ts=ob.ts,
        bids=cumulative_depth(ob.bids),
        asks=cumulative_depth(ob.asks),
    )


@api_router.get("/bitmex/analytics/snapshot", response_model=AnalyticsSnapshotResponse, tags=["analytics"])
async def analytics_snapshot(symbol: str, depth: int = 50, bands_bps: str = "10,25,100"):
    try:
        bands = [int(x.strip()) for x in bands_bps.split(",") if x.strip()]
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid bands_bps") from e

    l2 = bitmex_get("/orderBook/L2", params={"symbol": symbol, "depth": depth})
    metrics = compute_orderbook_metrics(l2=l2, bands_bps=bands)
    return AnalyticsSnapshotResponse(
        symbol=symbol,
        ts=iso(now_utc()),
        best_bid=metrics["best_bid"],
        best_ask=metrics["best_ask"],
        mid=metrics["mid"],
        spread=metrics["spread"],
        bands=metrics["bands"],
    )


@api_router.get("/bitmex/analytics/flow", response_model=FlowResponse, tags=["analytics"])
async def analytics_flow(symbol: str, minutes: int = 5):
    minutes = max(1, min(minutes, 60))
    count = min(1000, minutes * 200)  # heuristic
    trades = bitmex_get(
        "/trade",
        params={
            "symbol": symbol,
            "count": count,
            "reverse": "true",
        },
    )

    if not isinstance(trades, list) or not trades:
        raise HTTPException(status_code=502, detail="No trades from BitMEX")

    end_ts = parse_iso(trades[0]["timestamp"])
    start_ts = end_ts - timedelta(minutes=minutes)
    window = [t for t in trades if parse_iso(t["timestamp"]) >= start_ts]
    window = list(reversed(window))  # chronological

    if len(window) < 2:
        raise HTTPException(status_code=502, detail="Not enough trades in window")

    buy_vol = 0.0
    sell_vol = 0.0
    cvd = 0.0
    first_price = float(window[0].get("price") or 0)
    last_price = float(window[-1].get("price") or 0)

    for t in window:
        sz = float(t.get("size") or 0)
        side = t.get("side")
        if side == "Buy":
            buy_vol += sz
            cvd += sz
        elif side == "Sell":
            sell_vol += sz
            cvd -= sz

    total = buy_vol + sell_vol
    aggressive_imbalance = ((buy_vol - sell_vol) / total) if total else 0.0
    price_change = (last_price - first_price) if first_price else 0.0
    absorption_ratio = (total / (abs(price_change) + 1e-9))

    return FlowResponse(
        symbol=symbol,
        ts=iso(now_utc()),
        minutes=minutes,
        buy_volume=buy_vol,
        sell_volume=sell_vol,
        aggressive_imbalance=aggressive_imbalance,
        cvd=cvd,
        price_change=price_change,
        absorption_ratio=absorption_ratio,
    )


@api_router.get("/bitmex/flow/timeseries", response_model=FlowSeriesResponse, tags=["analytics"])
async def flow_timeseries(symbol: str, minutes: int = 60):
    minutes = max(5, min(minutes, 240))

    # BitMEX trade table has max count=1000; keep within limits.
    count = min(1000, minutes * 250)
    trades = bitmex_get(
        "/trade",
        params={
            "symbol": symbol,
            "count": count,
            "reverse": "true",
        },
    )
    if not isinstance(trades, list) or not trades:
        raise HTTPException(status_code=502, detail="No trades from BitMEX")

    end_ts = parse_iso(trades[0]["timestamp"])
    start_ts = end_ts - timedelta(minutes=minutes)

    # build per-minute bins
    bins: Dict[datetime, Dict[str, float]] = {}
    # iterate chronological
    for t in reversed(trades):
        ts = parse_iso(t["timestamp"])
        if ts < start_ts:
            continue
        minute = floor_to_minute(ts)
        if minute not in bins:
            bins[minute] = {"buy": 0.0, "sell": 0.0}
        sz = float(t.get("size") or 0)
        side = t.get("side")
        if side == "Buy":
            bins[minute]["buy"] += sz
        elif side == "Sell":
            bins[minute]["sell"] += sz

    # candles for close overlay
    candle_rows = bitmex_get(
        "/trade/bucketed",
        params={
            "symbol": symbol,
            "binSize": "1m",
            "partial": "false",
            "reverse": "false",
            "startTime": iso(start_ts),
            "endTime": iso(end_ts),
            "count": 500,
        },
    )
    close_by_minute: Dict[datetime, float] = {}
    if isinstance(candle_rows, list):
        for r in candle_rows:
            if not r.get("timestamp"):
                continue
            m = floor_to_minute(parse_iso(r["timestamp"]))
            close_by_minute[m] = float(r.get("close") or 0)

    minutes_list = sorted(bins.keys())
    points: List[FlowSeriesPoint] = []
    cvd = 0.0
    for m in minutes_list:
        buy = bins[m]["buy"]
        sell = bins[m]["sell"]
        delta = buy - sell
        cvd += delta
        points.append(
            FlowSeriesPoint(
                t=iso(m),
                buy=buy,
                sell=sell,
                delta=delta,
                cvd=cvd,
                close=close_by_minute.get(m),
            )
        )

    return FlowSeriesResponse(symbol=symbol, ts=iso(now_utc()), minutes=minutes, points=points)


@api_router.get("/bitmex/funding", response_model=FundingResponse, tags=["bitmex"])
async def bitmex_funding(symbol: str, start: str, end: str):
    start_dt = parse_iso(start)
    end_dt = parse_iso(end)
    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end must be after start")

    rows = bitmex_get(
        "/funding",
        params={
            "symbol": symbol,
            "startTime": iso(start_dt),
            "endTime": iso(end_dt),
            "count": 500,
            "reverse": "false",
        },
    )

    if not isinstance(rows, list):
        raise HTTPException(status_code=502, detail="Unexpected funding response")

    prev: Optional[float] = None
    points: List[FundingPoint] = []
    for r in rows:
        t = r.get("timestamp") or r.get("fundingTimestamp")
        if not t:
            continue
        fr = float(r.get("fundingRate") or 0)
        mom = fr - prev if prev is not None else 0.0
        points.append(FundingPoint(t=t, funding_rate=fr, momentum=mom))
        prev = fr

    return FundingResponse(symbol=symbol, ts=iso(now_utc()), points=points)


@api_router.get("/bitmex/open-interest", response_model=OpenInterestResponse, tags=["bitmex"])
async def bitmex_open_interest(symbol: str, start: str, end: str):
    # BitMEX may not expose the historical /openInterest table on mainnet.
    # Fallback: derive a time series from instrument snapshots (openInterest field).
    start_dt = parse_iso(start)
    end_dt = parse_iso(end)
    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end must be after start")

    # sample interval based on requested range
    total_minutes = max(1, int((end_dt - start_dt).total_seconds() / 60))
    if total_minutes <= 120:
        step = 5
    elif total_minutes <= 24 * 60:
        step = 15
    else:
        step = 60

    points_raw: List[Tuple[datetime, float]] = []

    cursor = floor_to_minute(start_dt)
    while cursor <= end_dt:
        # query instrument at/around timestamp
        rows = bitmex_get(
            "/instrument",
            params={
                "symbol": symbol,
                "count": 1,
                "reverse": "true",
                "startTime": iso(cursor),
                "endTime": iso(cursor + timedelta(minutes=step)),
            },
        )
        if isinstance(rows, list) and rows:
            r = rows[0]
            t = r.get("timestamp")
            oi = r.get("openInterest")
            if t is not None and oi is not None:
                points_raw.append((parse_iso(t), float(oi)))

        cursor = cursor + timedelta(minutes=step)
        if len(points_raw) > 600:
            break

    # dedupe & sort
    points_raw.sort(key=lambda x: x[0])
    dedup: List[Tuple[datetime, float]] = []
    last_t: Optional[datetime] = None
    for t, oi in points_raw:
        if last_t and t == last_t:
            continue
        dedup.append((t, oi))
        last_t = t

    prev: Optional[float] = None
    points: List[OpenInterestPoint] = []
    for t, oi in dedup:
        d = oi - prev if prev is not None else 0.0
        points.append(OpenInterestPoint(t=iso(t), open_interest=oi, delta=d))
        prev = oi

    return OpenInterestResponse(symbol=symbol, ts=iso(now_utc()), points=points)


@api_router.get("/bitmex/liquidations", response_model=LiquidationsResponse, tags=["bitmex"])
async def bitmex_liquidations(symbol: str, minutes: int = 60):
    minutes = max(5, min(minutes, 240))
    count = min(500, minutes * 50)

    rows = bitmex_get(
        "/liquidation",
        params={
            "symbol": symbol,
            "count": count,
            "reverse": "true",
        },
    )

    if not isinstance(rows, list):
        raise HTTPException(status_code=502, detail="Unexpected liquidation response")

    # Filter by time window if timestamps exist
    points: List[LiquidationPoint] = []
    end_ts = now_utc()
    start_ts = end_ts - timedelta(minutes=minutes)

    for r in rows:
        t = r.get("timestamp")
        if not t:
            continue
        ts = parse_iso(t)
        if ts < start_ts:
            continue
        points.append(
            LiquidationPoint(
                t=t,
                price=float(r.get("price") or 0),
                size=float(r.get("leavesQty") or r.get("orderQty") or r.get("qty") or 0),
                side=r.get("side"),
            )
        )

    points = list(reversed(points))
    return LiquidationsResponse(symbol=symbol, ts=iso(now_utc()), minutes=minutes, points=points)


# -------- Strategy CRUD --------
@api_router.post("/strategies", response_model=StrategyOut, tags=["strategies"])
async def create_strategy(payload: StrategyCreate, user: Dict[str, Any] = Depends(get_current_user)):
    doc = {
        "user_id": str(user["_id"]),
        "name": payload.name,
        "symbol": payload.symbol,
        "entry_conditions": [c.model_dump() for c in payload.entry_conditions],
        "exit_conditions": [c.model_dump() for c in payload.exit_conditions],
        "fee_bps": payload.fee_bps,
        "slippage_bps": payload.slippage_bps,
        "created_at": iso(now_utc()),
    }
    res = await db.strategies.insert_one(doc)
    return StrategyOut(id=str(res.inserted_id), **doc)


@api_router.get("/strategies", response_model=List[StrategyOut], tags=["strategies"])
async def list_strategies(user: Dict[str, Any] = Depends(get_current_user)):
    cur = db.strategies.find(
        {"user_id": str(user["_id"])},
        {
            "_id": 1,
            "user_id": 1,
            "name": 1,
            "symbol": 1,
            "entry_conditions": 1,
            "exit_conditions": 1,
            "fee_bps": 1,
            "slippage_bps": 1,
            "created_at": 1,
        },
    )
    docs = await cur.sort("created_at", -1).to_list(200)
    out: List[StrategyOut] = []
    for d in docs:
        d["id"] = str(d.pop("_id"))
        out.append(StrategyOut(**d))
    return out


@api_router.get("/strategies/{strategy_id}", response_model=StrategyOut, tags=["strategies"])
async def get_strategy(strategy_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    if not ObjectId.is_valid(strategy_id):
        raise HTTPException(status_code=400, detail="Invalid strategy id")
    d = await db.strategies.find_one({"_id": ObjectId(strategy_id), "user_id": str(user["_id"])})
    if not d:
        raise HTTPException(status_code=404, detail="Strategy not found")
    d["id"] = str(d.pop("_id"))
    return StrategyOut(**d)


@api_router.delete("/strategies/{strategy_id}", tags=["strategies"])
async def delete_strategy(strategy_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    if not ObjectId.is_valid(strategy_id):
        raise HTTPException(status_code=400, detail="Invalid strategy id")
    res = await db.strategies.delete_one({"_id": ObjectId(strategy_id), "user_id": str(user["_id"])})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {"ok": True}


# -------- Backtesting --------

def compute_drawdown(equity: List[float]) -> float:
    peak = -1e18
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


@api_router.post("/backtests/run", response_model=BacktestRunResponse, tags=["backtests"])
async def run_backtest(payload: BacktestRunRequest, user: Dict[str, Any] = Depends(get_current_user)):
    candles = await bitmex_candles(symbol=payload.symbol, start=payload.start, end=payload.end)
    if len(candles) < 200:
        raise HTTPException(status_code=400, detail="Not enough candles for backtest (need at least ~200 minutes)")

    series = compute_series(candles)

    fee = payload.strategy.fee_bps / 10000.0
    slip = payload.strategy.slippage_bps / 10000.0

    in_pos = False
    entry_price = 0.0
    entry_time = ""

    equity = payload.initial_capital
    equity_curve: List[Dict[str, Any]] = []
    trades: List[BacktestTrade] = []

    for i in range(0, len(candles) - 1):
        ts = candles[i].timestamp
        px = candles[i].close

        equity_curve.append({"t": ts, "equity": equity, "price": px})

        if not in_pos:
            if payload.strategy.entry_conditions and all(
                eval_condition(series, i, c) for c in payload.strategy.entry_conditions
            ):
                nxt = candles[i + 1]
                entry_time = nxt.timestamp
                entry_price = nxt.open * (1 + slip)
                equity *= (1 - fee)
                in_pos = True
        else:
            if payload.strategy.exit_conditions and all(
                eval_condition(series, i, c) for c in payload.strategy.exit_conditions
            ):
                nxt = candles[i + 1]
                exit_time = nxt.timestamp
                exit_price = nxt.open * (1 - slip)
                equity *= (1 - fee)

                ret = (exit_price / entry_price) - 1.0 if entry_price else 0.0

                before = equity
                equity = equity * (1 + ret)
                pnl = equity - before

                trades.append(
                    BacktestTrade(
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=exit_time,
                        exit_price=exit_price,
                        pnl=pnl,
                        return_pct=ret * 100,
                    )
                )
                in_pos = False

    equity_curve.append({"t": candles[-1].timestamp, "equity": equity, "price": candles[-1].close})

    total_return = (equity / payload.initial_capital - 1.0) if payload.initial_capital else 0.0
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = (wins / len(trades)) if trades else 0.0
    dd = compute_drawdown([p["equity"] for p in equity_curve])

    summary = BacktestSummary(
        total_return_pct=total_return * 100,
        max_drawdown_pct=dd * 100,
        win_rate_pct=win_rate * 100,
        trades=len(trades),
    )

    run_doc = {
        "user_id": str(user["_id"]),
        "created_at": iso(now_utc()),
        "symbol": payload.symbol,
        "start": payload.start,
        "end": payload.end,
        "strategy": payload.strategy.model_dump(),
        "initial_capital": payload.initial_capital,
        "summary": summary.model_dump(),
        "equity_curve": equity_curve,
        "trades": [t.model_dump() for t in trades],
    }

    res = await db.backtest_runs.insert_one(run_doc)

    return BacktestRunResponse(
        id=str(res.inserted_id),
        created_at=run_doc["created_at"],
        symbol=run_doc["symbol"],
        start=run_doc["start"],
        end=run_doc["end"],
        summary=summary,
        equity_curve=equity_curve,
        trades=trades,
    )


@api_router.get("/backtests", tags=["backtests"])
async def list_backtests(user: Dict[str, Any] = Depends(get_current_user)):
    cur = db.backtest_runs.find(
        {"user_id": str(user["_id"])},
        {"_id": 1, "created_at": 1, "symbol": 1, "start": 1, "end": 1, "summary": 1},
    )
    docs = await cur.sort("created_at", -1).to_list(200)
    for d in docs:
        d["id"] = str(d.pop("_id"))
    return docs


@api_router.get("/backtests/{run_id}", tags=["backtests"])
async def get_backtest(run_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    if not ObjectId.is_valid(run_id):
        raise HTTPException(status_code=400, detail="Invalid run id")
    d = await db.backtest_runs.find_one({"_id": ObjectId(run_id), "user_id": str(user["_id"])}, {"_id": 0})
    if not d:
        raise HTTPException(status_code=404, detail="Backtest not found")
    d["id"] = run_id
    return d


app.include_router(api_router)


@app.on_event("startup")
async def startup():
    # Store rolling 10 minutes of orderbook snapshots for a real time-based heatmap.
    async def persist_orderbook(ob: Dict[str, Any]):
        try:
            symbol = ob.get("symbol")
            if not symbol:
                return

            ts = ob.get("ts")
            if not ts:
                return

            doc = {
                "symbol": symbol,
                "ts": ts,
                "bids": ob.get("bids") or [],
                "asks": ob.get("asks") or [],
            }
            await db.orderbook_heat.insert_one(doc)

            cutoff = iso(now_utc() - timedelta(minutes=10))
            await db.orderbook_heat.delete_many({"symbol": symbol, "ts": {"$lt": cutoff}})
        except Exception as e:
            logger.debug("persist_orderbook failed: %s", e)

    ws_manager.on_orderbook = persist_orderbook

    # Start WS manager for default symbol.
    try:
        await ws_manager.start()
    except Exception as e:
        logger.warning("WS manager failed to start on startup: %s", e)


@app.on_event("shutdown")
async def shutdown_db_client():
    try:
        await ws_manager.stop()
    except Exception:
        pass
    client.close()
