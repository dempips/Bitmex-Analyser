import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export default function Register({ onRegister }) {
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
      await onRegister(email, password);
      nav("/app/analytics");
    } catch (err) {
      const msg = err?.response?.data?.detail || "Registration failed";
      setError(msg);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div data-testid="register-page" className="min-h-screen bg-[radial-gradient(900px_500px_at_10%_-10%,hsl(var(--chart-4))_0%,transparent_55%),radial-gradient(800px_500px_at_85%_0%,hsl(var(--chart-5))_0%,transparent_55%)]">
      <div className="mx-auto max-w-6xl px-6 py-14">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start">
          <div className="space-y-4">
            <h1 data-testid="register-title" className="text-4xl sm:text-5xl lg:text-6xl font-semibold tracking-tight">
              Create your workspace.
            </h1>
            <p data-testid="register-subtitle" className="text-base md:text-lg text-muted-foreground max-w-xl">
              Save strategies, run backtests, and keep your research organized.
            </p>

            <div data-testid="register-note" className="rounded-2xl border bg-card/70 p-4 backdrop-blur text-sm text-muted-foreground">
              You can start with public BitMEX data only. No API keys required.
            </div>
          </div>

          <Card data-testid="register-card" className="rounded-2xl shadow-sm">
            <CardHeader>
              <CardTitle data-testid="register-form-title" className="text-xl">Sign up</CardTitle>
              <CardDescription data-testid="register-form-desc">Email + password (JWT).</CardDescription>
            </CardHeader>
            <CardContent>
              {error ? (
                <Alert data-testid="register-error-alert" variant="destructive" className="mb-4">
                  <AlertTitle data-testid="register-error-title">Couldn’t create account</AlertTitle>
                  <AlertDescription data-testid="register-error-desc">{error}</AlertDescription>
                </Alert>
              ) : null}

              <form onSubmit={submit} className="space-y-4">
                <div className="space-y-2">
                  <Label data-testid="register-email-label" htmlFor="email">Email</Label>
                  <Input
                    data-testid="register-email-input"
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
                  <Label data-testid="register-password-label" htmlFor="password">Password</Label>
                  <Input
                    data-testid="register-password-input"
                    id="password"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    autoComplete="new-password"
                    required
                  />
                  <div data-testid="register-password-hint" className="text-xs text-muted-foreground">
                    Minimum 8 characters.
                  </div>
                </div>

                <Button
                  data-testid="register-submit-button"
                  type="submit"
                  className="w-full rounded-full"
                  disabled={busy}
                >
                  {busy ? "Creating…" : "Create account"}
                </Button>
              </form>

              <div className="mt-4 text-sm text-muted-foreground">
                <span data-testid="register-login-text">Already have an account?</span>{" "}
                <Link data-testid="register-login-link" className="text-foreground underline underline-offset-4" to="/login">
                  Sign in
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
