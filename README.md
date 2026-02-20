# BitMEX Analyser

A **BitMEX microstructure analytics and backtesting** web app. It provides a charts-first view of order book depth, flow, CVD (cumulative volume delta), funding, and open interest, with rule-based backtesting on 1-minute candles.

## What it is

- **Analytics dashboard** — Live and historical context: order book structure (depth, heatmap, OBI by band), aggressive buy/sell flow, CVD, funding rate, and open interest. Timestamps can be shown in UTC or local time.
- **Price chart** — OHLC candlesticks with optional volume and CVD overlays, configurable timeframe (1m–1w) and range (7d to max). Volume/CVD history is configurable (e.g. up to 365 days).
- **Backtesting** — Long-only, rule-based strategy builder using 1m candles. Entry/exit conditions can use price (close, returns, SMA/EMA, volatility) or signals derived from the analytics (liquidity imbalance, aggression imbalance, absorption score). Fee and slippage are applied; optional risk controls (stop loss, take profit, max hold).

## Tech stack

- **Frontend:** React (CRACO), Tailwind, Recharts, Lightweight Charts.
- **Backend:** Python (FastAPI), BitMEX REST and WebSocket for live order book and trades.

## Run locally

1. **Backend:** `cd backend && python -m venv venv && source venv/bin/activate` (or `venv\Scripts\activate` on Windows), `pip install -r requirements.txt`, then `uvicorn server:app --reload --host 0.0.0.0 --port 8000`.
2. **Frontend:** `cd frontend && npm install && npm start` (dev server on port 3000).
3. Or use the script: `bash scripts/kill-and-restart.sh` to clear ports and start both.

API base URL for the frontend is configured to the backend (e.g. `http://localhost:8000`). Sign in with email/password or use **Continue as guest** to try the app.
