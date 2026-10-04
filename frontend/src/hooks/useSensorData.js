import { useState, useEffect, useRef, useCallback } from "react";
import {
  getLiveSensorData,
  getSensorWebSocketUrl,
} from "../services/sensorService";

const DEFAULT_SENSORS = {
  ppg_ir: null,
  ppg_red: null,
  ecg: null,
  accel_x: null,
  accel_y: null,
  accel_z: null,
  gyro_x: null,
  gyro_y: null,
  gyro_z: null,
  temperature_c: null,
  piezo: null,
};

const DEFAULT_STATUS = {
  connected: false,
  connection_status: "disconnected",
  device_id: null,
  last_seen: null,
  sequence: 0,
  packet_count: 0,
  packet_loss: 0,
};

export function useSensorData() {
  const [sensorData, setSensorData] = useState(DEFAULT_SENSORS);
  const [status, setStatus] = useState(DEFAULT_STATUS);
  const [history, setHistory] = useState([]);
  const [backendAvailable, setBackendAvailable] = useState(true);
  const [isWebSocketActive, setIsWebSocketActive] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const pollIntervalRef = useRef(null);
  const isMountedRef = useRef(true);
  const lastSeenRef = useRef(null);

  // Compute live freshness state based on last_seen timestamp
  const calculateConnectionStatus = useCallback((lastSeenTimestamp, baseConnected) => {
    if (!lastSeenTimestamp) return "disconnected";
    const nowInSeconds = Date.now() / 1000;
    const age = nowInSeconds - Number(lastSeenTimestamp);
    if (age < 2.5) return "connected";
    if (age < 5.0) return "stale";
    return "disconnected";
  }, []);

  // Update raw and calculated state from document
  const applySensorDocument = useCallback((doc) => {
    if (!doc || typeof doc !== "object") return;

    if (doc.sensors) {
      setSensorData(doc.sensors);
    } else if (doc.latest?.sensors) {
      setSensorData(doc.latest.sensors);
    }

    if (doc.history && Array.isArray(doc.history)) {
      setHistory(doc.history);
    }

    if (doc.status) {
      const lastSeen = doc.status.last_seen;
      lastSeenRef.current = lastSeen;
      const connStatus = calculateConnectionStatus(lastSeen, doc.status.connected);

      setStatus({
        ...doc.status,
        connection_status: connStatus,
      });
    }

    setBackendAvailable(true);
    setError(null);
    setLoading(false);
  }, [calculateConnectionStatus]);

  // REST fetch fallback
  const fetchLiveRest = useCallback(async () => {
    try {
      const data = await getLiveSensorData();
      if (isMountedRef.current) {
        applySensorDocument(data);
      }
    } catch (err) {
      if (isMountedRef.current) {
        setBackendAvailable(false);
        setError("Sensor backend unavailable");
        setStatus((prev) => ({ ...prev, connection_status: "disconnected", connected: false }));
        setLoading(false);
      }
    }
  }, [applySensorDocument]);

  // Polling fallback management
  const startPolling = useCallback(() => {
    if (pollIntervalRef.current) return;
    fetchLiveRest();
    pollIntervalRef.current = setInterval(() => {
      if (isMountedRef.current) {
        fetchLiveRest();
      }
    }, 1500);
  }, [fetchLiveRest]);

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  // WebSocket connection handler
  const connectWebSocket = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.CONNECTING || wsRef.current.readyState === WebSocket.OPEN)) {
      return;
    }

    const wsUrl = getSensorWebSocketUrl();

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!isMountedRef.current) return;
        setIsWebSocketActive(true);
        setBackendAvailable(true);
        setError(null);
        // WebSocket is alive, halt HTTP polling
        stopPolling();
      };

      ws.onmessage = (event) => {
        if (!isMountedRef.current) return;
        try {
          const message = JSON.parse(event.data);
          applySensorDocument(message);
        } catch (parseErr) {
          console.warn("[Serenova WS] Failed to parse sensor payload:", parseErr);
        }
      };

      ws.onerror = (err) => {
        if (!isMountedRef.current) return;
        setIsWebSocketActive(false);
        // Fall back to polling immediately on WS error
        startPolling();
      };

      ws.onclose = () => {
        if (!isMountedRef.current) return;
        setIsWebSocketActive(false);
        // Fall back to polling and schedule reconnect
        startPolling();
        if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = setTimeout(() => {
          if (isMountedRef.current) {
            connectWebSocket();
          }
        }, 3000);
      };
    } catch (e) {
      setIsWebSocketActive(false);
      startPolling();
    }
  }, [applySensorDocument, startPolling, stopPolling]);

  // Periodic local freshness check (so status shifts from connected -> stale -> disconnected even without incoming frames)
  useEffect(() => {
    const freshnessTimer = setInterval(() => {
      if (lastSeenRef.current) {
        const computedStatus = calculateConnectionStatus(lastSeenRef.current, true);
        setStatus((prev) => {
          if (prev.connection_status !== computedStatus) {
            return {
              ...prev,
              connection_status: computedStatus,
              connected: computedStatus === "connected",
            };
          }
          return prev;
        });
      }
    }, 500);

    return () => clearInterval(freshnessTimer);
  }, [calculateConnectionStatus]);

  // Lifecycle
  useEffect(() => {
    isMountedRef.current = true;

    // 1. Initial snapshot fetch
    fetchLiveRest();

    // 2. Connect WebSocket
    connectWebSocket();

    return () => {
      isMountedRef.current = false;
      stopPolling();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [fetchLiveRest, connectWebSocket, stopPolling]);

  return {
    sensorData,
    status,
    history,
    connected: status.connection_status === "connected",
    connectionStatus: status.connection_status || "disconnected",
    lastSeen: status.last_seen,
    sequence: status.sequence || 0,
    packetCount: status.packet_count || 0,
    packetLoss: status.packet_loss || 0,
    backendAvailable,
    isWebSocketActive,
    error,
    loading,
    refetch: fetchLiveRest,
  };
}
