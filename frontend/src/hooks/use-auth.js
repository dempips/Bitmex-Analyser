import { useEffect, useMemo, useState } from "react";
import { api, clearToken, getToken, setToken } from "@/api/client";

export function useAuth() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const isAuthed = useMemo(() => !!getToken(), []);

  async function fetchMe() {
    const token = getToken();
    if (!token) {
      setUser(null);
      setLoading(false);
      return;
    }
    try {
      const res = await api.get("/auth/me", { timeout: 10000 });
      setUser(res.data.user);
    } catch (e) {
      clearToken();
      setUser(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    let cancelled = false;
    const done = () => {
      if (!cancelled) setLoading(false);
    };
    const t = setTimeout(done, 12000);
    fetchMe().finally(() => {
      clearTimeout(t);
      done();
    });
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, []);

  async function login(email, password) {
    const res = await api.post("/auth/login", { email, password });
    setToken(res.data.token);
    setUser(res.data.user);
    return res.data;
  }

  async function guestLogin() {
    const res = await api.post("/auth/guest");
    setToken(res.data.token);
    setUser(res.data.user);
    return res.data;
  }

  async function register(email, password) {
    const res = await api.post("/auth/register", { email, password });
    setToken(res.data.token);
    setUser(res.data.user);
    return res.data;
  }

  function logout() {
    clearToken();
    setUser(null);
  }

  return { user, loading, isAuthed, login, guestLogin, register, logout, refresh: fetchMe };
}
