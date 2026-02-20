import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export default function Login({ onLogin, onGuest }) {
  const nav = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await onLogin(email, password);
      nav("/app/analytics");
    } catch (err) {
      const msg = err?.response?.data?.detail || "Login failed";
      setError(msg);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div data-testid="login-page" className="min-h-screen bg-[radial-gradient(1200px_500px_at_20%_-10%,hsl(var(--chart-3))_0%,transparent_55%),radial-gradient(900px_500px_at_90%_0%,hsl(var(--chart-2))_0%,transparent_50%)]">
      <div className="mx-auto max-w-6xl px-6 py-14">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start">
          <div className="space-y-6">
            <div className="space-y-3">
              <h1 data-testid="login-hero-title" className="text-4xl sm:text-5xl lg:text-6xl font-semibold tracking-tight">
                BitMEX microstructure,
                <span className="text-muted-foreground"> made readable</span>.
              </h1>
              <p data-testid="login-hero-subtitle" className="text-base md:text-lg text-muted-foreground max-w-xl">
                TradeMetryx is an analytics + backtesting workspace built around BitMEX mechanics: order-book structure,
                flow, and regime context.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div data-testid="login-feature-1" className="rounded-2xl border bg-card/70 p-4 backdrop-blur">
                <div className="text-sm font-medium">Order book imbalance</div>
                <div className="text-sm text-muted-foreground">Depth-weighted OBI over configurable ranges.</div>
              </div>
              <div data-testid="login-feature-2" className="rounded-2xl border bg-card/70 p-4 backdrop-blur">
                <div className="text-sm font-medium">Flow metrics</div>
                <div className="text-sm text-muted-foreground">CVD + aggression and absorption proxy.</div>
              </div>
              <div data-testid="login-feature-3" className="rounded-2xl border bg-card/70 p-4 backdrop-blur">
                <div className="text-sm font-medium">Backtesting</div>
                <div className="text-sm text-muted-foreground">Rule-based strategy builder (1m candles).</div>
              </div>
              <div data-testid="login-feature-4" className="rounded-2xl border bg-card/70 p-4 backdrop-blur">
                <div className="text-sm font-medium">BitMEX-only</div>
                <div className="text-sm text-muted-foreground">No external feeds. No mixed-market noise.</div>
              </div>
            </div>
          </div>

          <Card data-testid="login-card" className="rounded-2xl shadow-sm">
            <CardHeader>
              <CardTitle data-testid="login-form-title" className="text-xl">Sign in</CardTitle>
              <CardDescription data-testid="login-form-desc">Use your email + password.</CardDescription>
            </CardHeader>
            <CardContent>
              {error ? (
                <Alert data-testid="login-error-alert" variant="destructive" className="mb-4">
                  <AlertTitle data-testid="login-error-title">Couldn’t sign you in</AlertTitle>
                  <AlertDescription data-testid="login-error-desc">{error}</AlertDescription>
                </Alert>
              ) : null}

              <form onSubmit={submit} className="space-y-4">
                <div className="space-y-2">
                  <Label data-testid="login-email-label" htmlFor="email">Email</Label>
                  <Input
                    data-testid="login-email-input"
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@domain.com"
                    autoComplete="email"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label data-testid="login-password-label" htmlFor="password">Password</Label>
                  <Input
                    data-testid="login-password-input"
                    id="password"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    autoComplete="current-password"
                    required
                  />
                  <div data-testid="login-password-hint" className="text-xs text-muted-foreground">
                    Minimum 8 characters.
                  </div>
                </div>
                <Button
                  data-testid="login-submit-button"
                  type="submit"
                  className="w-full rounded-full"
                  disabled={busy}
                >
                  {busy ? "Signing in…" : "Sign in"}
                </Button>

                <Button
                  data-testid="login-guest-button"
                  type="button"
                  variant="outline"
                  className="w-full rounded-full"
                  disabled={busy}
                  onClick={async () => {
                    if (!onGuest) return;
                    setError(null);
                    setBusy(true);
                    try {
                      await onGuest();
                      // Defer navigation so auth state is committed before Protected runs
                      setTimeout(() => nav("/app/analytics"), 0);
                    } catch (err) {
                      const msg = err?.response?.data?.detail || "Guest login failed";
                      setError(msg);
                    } finally {
                      setBusy(false);
                    }
                  }}
                >
                  Continue as guest
                </Button>
              </form>

              <div className="mt-4 text-sm text-muted-foreground">
                <span data-testid="login-register-text">New here?</span>{" "}
                <Link data-testid="login-register-link" className="text-foreground underline underline-offset-4" to="/register">
                  Create an account
                </Link>
              </div>

              <div data-testid="login-security-note" className="mt-6 text-xs text-muted-foreground">
                Security note: MVP uses a server-side JWT token stored in your browser local storage.
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
