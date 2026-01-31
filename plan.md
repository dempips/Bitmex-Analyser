# TradeMetryx (BitMEX Analytics + Backtesting) — MVP Plan

## User Choices (Locked for MVP)
- Instruments: **Any BitMEX symbol** (user selects)
- Data mode: **REST-only** (near real-time via polling)
- Backtesting: **1-minute candles + derived metrics**
- Strategies: **Rule builder** (threshold conditions)
- Accounts: **Email+password (JWT)**

---

## 1) Architecture (MVP)
### Frontend (React + Tailwind + shadcn)
- Routes
  - `/login` — login
  - `/register` — create account
  - `/app` — authenticated app shell
    - Analytics tab (REST polling)
    - Backtesting tab (rule builder + results)
- API access via `process.env.REACT_APP_BACKEND_URL` only

### Backend (FastAPI)
- REST endpoints under `/api`
- JWT auth (email+password)
- BitMEX public REST proxy + analytics computations
- Backtesting engine (long-only for MVP)

### Database (MongoDB)
Collections:
- `users`
- `strategies`
- `backtest_runs`

---

## 2) Data Sources (BitMEX only)
Public BitMEX REST endpoints (server-side):
- Instruments/symbols: `GET /instrument/active`
- Order book snapshot: `GET /orderBook/L2?symbol=...&depth=...`
- Trades (for flow metrics): `GET /trade?symbol=...&count=...&reverse=true`
- 1m candles: `GET /trade/bucketed?symbol=...&binSize=1m&partial=false&reverse=false&startTime=...&endTime=...`
- Funding history (optional in MVP): `GET /funding?symbol=...`

We will **not** require BitMEX API keys for MVP.

---

## 3) MVP Analytics Metrics
### Order Book (snapshot-based)
- Best bid/ask, mid, spread
- Depth sums within configurable bps bands (default bands: 10, 25, 100 bps)
- **Order Book Imbalance (OBI)**
  - OBI = (bidDepth - askDepth) / (bidDepth + askDepth)
  - Enhanced: weighted by distance to mid within each band

### Trade Flow (recent window; computed from trades REST)
- Aggressive Buy/Sell imbalance (buy vs sell trade size)
- CVD (cumulative delta over the recent window)
- Absorption proxy: aggressive volume vs price change over window

### Volatility / Regime (candle-based)
- Micro-volatility compression proxy: rolling 1m return std
- Spread stability proxy (from snapshots, later; in MVP show only current spread)

### Composite Scores (decomposable, NOT buy/sell)
- Liquidity Defense Score (from OBI + depth)
- Aggression Efficiency Score (from absorption proxy)

---

## 4) Strategy Rule Builder (MVP)
### Supported Metrics (candle-derived)
- `close`
- `return_1` (close-to-close %)
- `sma_N` (N selectable: 10/20/50)
- `ema_N` (10/20/50)
- `volatility_N` (rolling std of returns; 10/20/50)

### Conditions Model
- Each condition: `{ metric, operator, value }`
- Strategy:
  - `entry_conditions`: all must be true (AND)
  - `exit_conditions`: all must be true (AND)
  - `position_size_pct` (e.g., 100% = fully invested)
  - `fee_bps`, `slippage_bps`

### Execution Model (MVP)
- Long-only
- Enter at **next bar open** when entry conditions become true
- Exit at **next bar open** when exit conditions become true
- Output:
  - trades list
  - equity curve
  - summary stats (total return, max drawdown, win rate, # trades)

---

## 5) Backend API (MVP)
### Auth
- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`

### BitMEX Data + Analytics
- `GET /api/bitmex/symbols`
- `GET /api/bitmex/candles?symbol=...&start=...&end=...`
- `GET /api/bitmex/analytics/snapshot?symbol=...`
- `GET /api/bitmex/analytics/flow?symbol=...&minutes=5`

### Strategies + Backtests (auth required)
- `POST /api/strategies`
- `GET /api/strategies`
- `GET /api/strategies/{id}`
- `DELETE /api/strategies/{id}`

- `POST /api/backtests/run` (runs and stores)
- `GET /api/backtests`
- `GET /api/backtests/{id}`

---

## 6) Frontend UX Flows (MVP)
1. Register/Login
2. Choose a symbol (searchable)
3. Analytics tab
   - Poll snapshot + flow metrics every ~5s
   - Show metric cards + small charts (candles / returns)
4. Backtesting tab
   - Create strategy with rule builder
   - Choose date range
   - Run backtest
   - View equity curve + trades table

---

## 7) Testing Approach
- Backend: curl tests
  - register/login/me
  - symbols
  - candles for a known symbol
  - run backtest
- Frontend: Playwright via testing agent
  - full auth flow
  - analytics loads
  - create + run backtest

---

## MVP Notes / Constraints
- REST-only means “real-time” is approximated via polling.
- Some BitMEX endpoints have pagination/limits; MVP will implement safe paging but may limit maximum date range per request.
