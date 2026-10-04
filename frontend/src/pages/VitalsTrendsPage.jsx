import React, { useMemo } from "react";
import {
  SectionHeader,
  Card,
  StatusBadge,
  PulsingDot,
  ApiResponseViewer,
} from "../components/common/CommonComponents";

/**
 * Simple, responsive SVG Sparkline for live hardware waveform buffers
 */
function SignalWaveform({ data = [], strokeColor = "#E05675", height = 70, minVal, maxVal, unit = "" }) {
  const points = useMemo(() => {
    const valid = data.filter((v) => v !== null && v !== undefined && !isNaN(v));
    if (valid.length < 2) return "";

    const autoMin = minVal !== undefined ? minVal : Math.min(...valid);
    const autoMax = maxVal !== undefined ? maxVal : Math.max(...valid);
    const range = autoMax - autoMin === 0 ? 1 : autoMax - autoMin;

    const width = 400;
    const step = width / (valid.length - 1);

    return valid
      .map((val, idx) => {
        const x = idx * step;
        const normalized = (val - autoMin) / range;
        const y = height - 8 - normalized * (height - 16);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }, [data, height, minVal, maxVal]);

  const latestVal = data.length > 0 ? data[data.length - 1] : "—";

  return (
    <div className="w-full bg-[#FAF7FD] border border-[#EAE4F5] rounded-xl p-3">
      <div className="flex justify-between items-center text-[12px] mb-1.5">
        <span className="text-muted font-medium">Telemetry Buffer ({data.length} pts)</span>
        <span className="font-bold text-ink" style={{ color: strokeColor }}>
          {latestVal !== null && latestVal !== undefined ? `${latestVal} ${unit}` : "—"}
        </span>
      </div>

      {points ? (
        <svg viewBox={`0 0 400 ${height}`} className="w-full h-16 overflow-visible">
          <polyline
            points={points}
            fill="none"
            stroke={strokeColor}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ) : (
        <div className="h-16 flex items-center justify-center text-[12px] text-muted italic">
          Waiting for live packet stream…
        </div>
      )}
    </div>
  );
}

export default function VitalsTrendsPage({ sensorState }) {
  const history = sensorState?.history || [];
  const status = sensorState?.status || {};
  const connStatus = sensorState?.connectionStatus || "disconnected";

  // Extract technical signal series from JSON history buffer
  const ecgSeries = history.map((h) => h.sensors?.ecg).filter((v) => v !== null && v !== undefined);
  const ppgIrSeries = history.map((h) => h.sensors?.ppg_ir).filter((v) => v !== null && v !== undefined);
  const ppgRedSeries = history.map((h) => h.sensors?.ppg_red).filter((v) => v !== null && v !== undefined);
  const tempSeries = history.map((h) => h.sensors?.temperature_c).filter((v) => v !== null && v !== undefined);
  const piezoSeries = history.map((h) => h.sensors?.piezo).filter((v) => v !== null && v !== undefined);

  return (
    <div className="space-y-5 animate-fade-up">
      {/* Header status bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 bg-white border border-border rounded-2xl">
        <div className="flex items-center gap-2">
          <span className="font-bold text-ink uppercase tracking-wider text-[11px]">
            Technical Sensor Telemetry:
          </span>
          <span
            className={`inline-flex items-center gap-1 font-semibold px-2.5 py-0.5 rounded-full text-[11px] ${
              connStatus === "connected"
                ? "bg-sage-light text-sage-dark"
                : connStatus === "stale"
                ? "bg-amber-light text-amber-dark"
                : "bg-rose-light text-rose-dark"
            }`}
          >
            <PulsingDot
              color={
                connStatus === "connected"
                  ? "sage"
                  : connStatus === "stale"
                  ? "amber"
                  : "rose"
              }
            />
            {connStatus.toUpperCase()} ({status.device_id || "serenova-pico-01"})
          </span>
        </div>
        <div className="text-[11px] text-muted">
          Packets logged in buffer: <strong>{history.length}</strong> · Total received: <strong>{status.packet_count || 0}</strong>
        </div>
      </div>

      {/* Real-time Hardware Waveform Grid */}
      <div className="grid gap-5 md:grid-cols-2">
        {/* ECG Signal */}
        <Card>
          <SectionHeader
            title="Live ECG Signal"
            right={<StatusBadge status="normal" label="AD8232 (Lead I)" />}
          />
          <p className="text-[12px] text-muted mb-3">
            Raw differential biopotential ADC amplitude from chest/wrist electrodes.
          </p>
          <SignalWaveform
            data={ecgSeries.slice(-60)}
            strokeColor="#E05675"
            unit="ADC"
          />
        </Card>

        {/* PPG Signal */}
        <Card>
          <SectionHeader
            title="Live PPG Signal (Photoplethysmogram)"
            right={<StatusBadge status="normal" label="MAX30102" />}
          />
          <p className="text-[12px] text-muted mb-3">
            Infrared & Red optical absorption waveform representing pulsatile blood volume.
          </p>
          <SignalWaveform
            data={ppgIrSeries.slice(-60)}
            strokeColor="#7052B8"
            unit="IR cnt"
          />
          {ppgRedSeries.length > 0 && (
            <div className="mt-2 text-[10px] text-muted flex justify-between px-1">
              <span>Red Channel: {ppgRedSeries[ppgRedSeries.length - 1] ?? "—"}</span>
              <span>Optical raw counts</span>
            </div>
          )}
        </Card>

        {/* Temperature Trend */}
        <Card>
          <SectionHeader
            title="Temperature Signal"
            right={<StatusBadge status="normal" label="DS18B20" />}
          />
          <p className="text-[12px] text-muted mb-3">
            Direct skin contact thermal sensor readings in degrees Celsius.
          </p>
          <SignalWaveform
            data={tempSeries.slice(-60)}
            strokeColor="#2A9968"
            unit="°C"
          />
        </Card>

        {/* Movement / Piezo Acoustic Signal */}
        <Card>
          <SectionHeader
            title="Movement & Piezo Acoustic Signal"
            right={<StatusBadge status="normal" label="Piezo Sensor" />}
          />
          <p className="text-[12px] text-muted mb-3">
            Acoustic contact vibrations and maternal/fetal mechanical movement amplitude.
          </p>
          <SignalWaveform
            data={piezoSeries.slice(-60)}
            strokeColor="#C48A08"
            unit="raw"
          />
        </Card>
      </div>

      {/* Raw history JSON payload viewer */}
      <ApiResponseViewer
        data={{
          telemetry_status: status,
          buffer_length: history.length,
          latest_record: history[history.length - 1] || null,
        }}
      />
    </div>
  );
}
