const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/**
 * Derives the WebSocket URL from the configured HTTP API base URL.
 * Supports http -> ws, https -> wss, and custom hosts/ports.
 */
export function getSensorWebSocketUrl() {
  const url = API_BASE_URL.trim();
  if (url.startsWith("https://")) {
    return url.replace(/^https:\/\//, "wss://") + "/api/sensor/ws";
  }
  if (url.startsWith("http://")) {
    return url.replace(/^http:\/\//, "ws://") + "/api/sensor/ws";
  }
  if (url.startsWith("ws://") || url.startsWith("wss://")) {
    return url.endsWith("/api/sensor/ws") ? url : `${url}/api/sensor/ws`;
  }
  // Fallback default
  return `ws://${window.location.hostname}:8000/api/sensor/ws`;
}

/**
 * Fetch latest live sensor data JSON snapshot from FastAPI.
 */
export async function getLiveSensorData() {
  const res = await fetch(`${API_BASE_URL}/api/sensor/live`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(`HTTP error ${res.status}: Failed to fetch live sensor data`);
  }
  return await res.json();
}

/**
 * Fetch high-level connection & telemetry status summary from FastAPI.
 */
export async function getSensorStatus() {
  const res = await fetch(`${API_BASE_URL}/api/sensor/status`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(`HTTP error ${res.status}: Failed to fetch sensor status`);
  }
  return await res.json();
}

/**
 * Fetch complete raw JSON record.
 */
export async function getSensorJson() {
  const res = await fetch(`${API_BASE_URL}/api/sensor/json`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(`HTTP error ${res.status}: Failed to fetch sensor json`);
  }
  return await res.json();
}

export { API_BASE_URL };
