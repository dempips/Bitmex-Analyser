import axios from "axios";

export const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || process.env.REACT_APP_API_URL || "http://localhost:8000";
const API_BASE = `${BACKEND_URL}/api`;

export function getToken() {
  return localStorage.getItem("tmx_token");
}

export function setToken(token) {
  localStorage.setItem("tmx_token", token);
}

export function clearToken() {
  localStorage.removeItem("tmx_token");
}

export const api = axios.create({
  baseURL: API_BASE,
});

api.interceptors.request.use((config) => {
  const t = getToken();
  if (t) {
    config.headers.Authorization = `Bearer ${t}`;
  }
  return config;
});
