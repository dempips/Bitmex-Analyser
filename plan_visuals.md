# TradeMetryx — Visualization Expansion Plan (All Selected)

## Goal
Add highly-visual charts for:
- Order book (depth + heatmap)
- Flow (CVD+price overlay + buy/sell bars)
- Timeframes (15m/60m/4h + custom)
- Backtesting (equity+drawdown + trade markers on price)
- BitMEX-specific (funding momentum + open interest delta + liquidation clusters)

## Backend Additions (MVP-friendly)
### New/extended endpoints
1) **Order book depth chart data**
- `GET /api/bitmex/orderbook/depth?symbol=...&depth=...`
  - Returns price levels for bids/asks and cumulative sizes (for depth chart)

2) **Order book heatmap (REST-polled snapshots)**
- `POST /api/bitmex/orderbook/heatmap/append`
  - Stores a snapshot in Mongo for a short rolling window (per user+symbol)
- `GET /api/bitmex/orderbook/heatmap?symbol=...&minutes=...`
  - Returns matrix-like series for price bins over time

3) **Flow time-series**
- `GET /api/bitmex/flow/timeseries?symbol=...&bin=1m&minutes=...`
  - Computes per-minute buyVol/sellVol and CVD series from trades

4) **Funding**
- `GET /api/bitmex/funding?symbol=...&start=...&end=...`
  - `fundingRate` series + momentum (delta)

5) **Open interest**
- `GET /api/bitmex/open-interest?symbol=...&start=...&end=...`
  - openInterest series + delta

6) **Liquidations (as proxy clusters)**
- `GET /api/bitmex/liquidations?symbol=...&minutes=...`
  - Uses `/api/v1/liquidation` recent feed
  - Build a simple “heat” by price bucket over time (probabilistic cluster proxy)

## Frontend Additions
### Analytics page
- Add a new **Visuals** section with tabs:
  1) Depth chart (bids/asks cumulative)
  2) Heatmap (order book liquidity vs time)
  3) Flow charts:
     - CVD + price overlay
     - Buy vs sell bars
  4) Funding + OI + Liquidations:
     - Funding line + momentum
     - OI line + delta
     - Liquidation bubble/heat scatter

### Backtest page
- Add a **Results** panel with:
  - Equity + drawdown (2-series or two charts)
  - Price chart with entry/exit markers (scatter overlay)

### Timeframes
- Add quick presets (15m/60m/4h) and a custom range picker

## Storage / Performance
- Heatmap will be **short rolling window** stored per user+symbol
- Use polling interval (e.g., 5s) to append snapshots
- Keep server-side aggregation lightweight; add safeguards for max points

## Testing
- Extend backend tests for new endpoints
- Frontend smoke test: load each visualization tab without errors
