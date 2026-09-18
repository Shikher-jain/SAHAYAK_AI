/**
 * Shared API client config for the Sahayak AI v2_advanced SPA.
 *
 * Every call funnels through one Axios instance so the base URL, the request
 * timeout, and the auth header are configured in exactly one place. In dev,
 * Vite proxies `/api` → http://127.0.0.1:8000 (see vite.config.js), so none of
 * the upstream host strings leak into component code.
 */
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || '/api'

export const apiBase = `${API_BASE}/v2`

/** One shared client — 60 s ceiling, JSON by default (multipart per-call). */
export const ax = axios.create({
  baseURL: apiBase,
  timeout: 60_000,
  headers: { 'Content-Type': 'application/json' },
})

/** POST the user turn to the supervisor; resolves to the v2 ChatResponse. */
export async function sendChat(text, userId = 'guest_user', tier = 'free') {
  const { data } = await ax.post('/chat', { message: text, user_id: userId, tier })
  return data
}
