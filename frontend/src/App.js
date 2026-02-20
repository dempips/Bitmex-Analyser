import React from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import "@/App.css";

import { useAuth } from "@/hooks/use-auth";

import Login from "@/pages/Login";
import Register from "@/pages/Register";
import Analytics from "@/pages/Analytics";
import Backtest from "@/pages/Backtest";
import AppShell from "@/components/layout/AppShell";

function Protected({ authed, loading, children }) {
  if (loading) {
    return (
      <div data-testid="app-loading" className="min-h-screen flex items-center justify-center">
        <div data-testid="app-loading-text" className="text-sm text-muted-foreground">Loading…</div>
      </div>
    );
  }
  if (!authed) return <Navigate to="/login" replace />;
  return children;
}

export default function App() {
  const auth = useAuth();

  return (
    <div data-testid="app-root" className="min-h-screen">
      <BrowserRouter>
        <Routes>
          <Route
            path="/"
            element={auth.user ? <Navigate to="/app/analytics" replace /> : <Navigate to="/login" replace />}
          />
          <Route path="/login" element={<Login onLogin={auth.login} onGuest={auth.guestLogin} />} />
          <Route path="/register" element={<Register onRegister={auth.register} />} />

          <Route
            path="/app"
            element={
              <Protected authed={!!auth.user} loading={auth.loading}>
                <AppShell user={auth.user} onLogout={auth.logout} />
              </Protected>
            }
          >
            <Route index element={<Navigate to="/app/analytics" replace />} />
            <Route path="analytics" element={<Analytics />} />
            <Route path="backtest" element={<Backtest />} />
          </Route>

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}
