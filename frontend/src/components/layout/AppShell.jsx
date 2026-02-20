import React from "react";
import { Link, NavLink, Outlet } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export default function AppShell({ user, onLogout }) {
  return (
    <div data-testid="app-shell" className="min-h-screen bg-background">
      <div className="border-b bg-card/70 backdrop-blur supports-[backdrop-filter]:bg-card/60">
        <div className="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <Link data-testid="nav-home-link" to="/app" className="flex items-baseline gap-2">
              <div className="text-lg font-semibold tracking-tight">TradeMetryx</div>
              <Badge data-testid="nav-beta-badge" variant="secondary" className="rounded-full">MVP</Badge>
            </Link>
            <div className="hidden md:flex items-center gap-2 text-sm text-muted-foreground">
              <span data-testid="nav-subtitle">BitMEX-only analytics + backtesting</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <nav className="hidden sm:flex items-center gap-1">
              <NavLink
                data-testid="nav-analytics-link"
                to="/app/analytics"
                className={({ isActive }) =>
                  `px-3 py-2 text-sm rounded-full transition-colors ${isActive ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground hover:bg-accent"}`
                }
              >
                Analytics
              </NavLink>
              <NavLink
                data-testid="nav-backtest-link"
                to="/app/backtest"
                className={({ isActive }) =>
                  `px-3 py-2 text-sm rounded-full transition-colors ${isActive ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground hover:bg-accent"}`
                }
              >
                Backtest
              </NavLink>
            </nav>

            <div className="hidden md:flex items-center gap-2 pl-2 border-l">
              <div data-testid="nav-user-email" className="text-sm text-muted-foreground">
                {user?.email}
              </div>
              <Button data-testid="logout-button" variant="outline" className="rounded-full" onClick={onLogout}>
                Logout
              </Button>
            </div>

            <div className="md:hidden">
              <Button data-testid="logout-button-mobile" variant="outline" className="rounded-full" onClick={onLogout}>
                Logout
              </Button>
            </div>
          </div>
        </div>
      </div>

      <main className="mx-auto max-w-7xl px-6 py-8">
        <Outlet />
      </main>

      <footer className="mx-auto max-w-7xl px-6 pb-10 text-xs text-muted-foreground">
        <div data-testid="footer-note">Data: BitMEX public REST. “Real-time” is approximated via polling.</div>
      </footer>
    </div>
  );
}
