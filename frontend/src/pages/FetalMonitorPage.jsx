import React, { useState, useEffect } from "react";
import {
  StatusBadge,
  ProgressBar,
  SectionHeader,
  Card,
  HighlightCard,
  PulsingDot,
  HeartRateWave,
  ApiResponseViewer,
} from "../components/common/CommonComponents";

function KickCounter() {
  const [kicks, setKicks] = useState(0);
  const [log, setLog] = useState([]);

  const addKick = () => {
    setKicks((k) => k + 1);
    setLog((l) => [
      ...l,
      new Date().toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
    ]);
  };

  const resetKicks = () => {
    setKicks(0);
    setLog([]);
  };

  const reached = kicks >= 10;

  return (
    <div className="text-center">
      <div
        className="font-display text-6xl font-bold leading-none mb-1 transition-colors duration-300"
        style={{ color: reached ? "#2A9968" : "#E05675" }}
      >
        {kicks}
      </div>
      <div className="text-[12px] text-muted mb-4">
        {kicks} / 10 kicks · goal in 2 hours{reached ? " ✓" : ""}
      </div>
      {reached && (
        <div className="mb-3 p-2.5 bg-sage-light border border-sage/25 rounded-xl text-[12px] font-semibold text-sage-dark">
          ✓ Target reached! Baby is very active.
        </div>
      )}
      <button
        onClick={addKick}
        className="w-full py-3.5 rounded-full bg-rose text-white font-semibold text-[15px] hover:bg-rose-dark active:scale-95 transition-all mb-2"
      >
        Feel a kick 👶
      </button>
      <button
        onClick={resetKicks}
        className="w-full py-2 rounded-full bg-rose-light text-rose text-[13px] font-semibold hover:bg-rose-medium transition-colors"
      >
        Reset
      </button>
      {log.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5 justify-center">
          {log.slice(-6).map((time, idx) => (
            <span
              key={idx}
              className="text-[10px] text-muted bg-surface px-2 py-0.5 rounded-full border border-border"
            >
              {time}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function FetalMonitorPage({ data, sensorState }) {
  const [fhrLive, setFhrLive] = useState(data.fetal.fhr);

  const sensorData = sensorState?.sensorData || {};
  const status = sensorState?.status || {};
  const connStatus = sensorState?.connectionStatus || "disconnected";

  useEffect(() => {
    const timer = setInterval(() => {
      setFhrLive((prev) =>
        Math.max(118, Math.min(162, prev + Math.round((Math.random() - 0.5) * 6)))
      );
    }, 2400);
    return () => clearInterval(timer);
  }, []);

  const hourlyKicks = [
    15, 12, 23, 18, 7, 20, 17, 14, 9, 11, 22, 19, 8, 13, 21, 16, 10, 18, 5, 12,
    20, 15, 11, 17,
  ];

  return (
    <div className="space-y-5 animate-fade-up">
      {/* Live Pico W Hardware Telemetry Banner */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-white border border-border rounded-2xl">
        <div className="flex items-center gap-2 text-[12px]">
          <span className="font-bold text-ink text-[11px] uppercase tracking-wider">
            Belt & Wrist Telemetry:
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
          Piezo Acoustic: <strong>{sensorData.piezo ?? "—"}</strong> · Temp:{" "}
          <strong>{sensorData.temperature_c ?? "—"}°C</strong>
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        {/* Left Column */}
        <div className="space-y-4">
          {/* Live Monitoring Box (CTG/NST Baseline) */}
          <div
            className="rounded-2xl p-5"
            style={{
              background: "linear-gradient(135deg,#FFF0F4,#FFD6E0)",
              border: "1px solid #FFD6E0",
            }}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <PulsingDot color="rose" />
                <span className="text-[11px] font-semibold text-rose uppercase tracking-wider">
                  Live CTG Monitoring
                </span>
              </div>
              <span className="text-[10px] uppercase font-bold text-rose-dark bg-white/60 px-2 py-0.5 rounded-full">
                Simulated Baseline
              </span>
            </div>
            <div className="text-[11px] font-semibold text-rose-dark mb-1">
              Fetal Heart Rate
            </div>
            <div className="font-display text-4xl font-bold text-rose leading-none mb-0.5">
              {fhrLive}
              <span className="text-[16px] font-body font-normal text-rose-dark ml-2">
                bpm
              </span>
            </div>
            <HeartRateWave />
            <div className="grid grid-cols-3 gap-2 mt-2">
              {[
                { label: "Variability", val: `${data.fetal.variability} bpm` },
                { label: "Kicks/hr", val: data.fetal.movement_count },
                { label: "Risk", val: "Low", color: "#2A9968" },
              ].map((item) => (
                <div key={item.label} className="bg-white rounded-xl py-2 text-center">
                  <div
                    className="font-display text-[18px] font-bold text-ink"
                    style={{ color: item.color }}
                  >
                    {item.val}
                  </div>
                  <div className="text-[10px] text-muted">{item.label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* CNN Fetal Hypoxia Card */}
          <Card>
            <SectionHeader title="CNN Fetal Hypoxia Analysis (Demo Model)" />
            <div className="flex items-center gap-4">
              <div className="text-center flex-shrink-0">
                <div className="font-display text-4xl font-bold text-sage">
                  {(data.fetal.cnn_raw.hypoxia_probability * 100).toFixed(0)}%
                </div>
                <div className="text-[11px] text-muted">Hypoxia probability</div>
              </div>
              <div className="flex-1">
                <ProgressBar
                  value={data.fetal.cnn_raw.hypoxia_probability * 100}
                  color="#2A9968"
                  className="mb-2"
                />
                <div className="text-[11px] text-sage font-semibold">
                  ✓ Below 30% threshold — no flag
                </div>
                <div className="text-[11px] text-muted mt-1">
                  Flag: {data.fetal.cnn_raw.hypoxia_flag ? "YES" : "No"} · Status:{" "}
                  {data.fetal.fetal_status}
                </div>
              </div>
            </div>
          </Card>

          {/* Kick Counter Card */}
          <Card>
            <SectionHeader
              title="Kick counter"
              right={<StatusBadge status="normal" label="Active" />}
            />
            <p className="text-[12px] text-muted mb-4">
              Tap each time you feel baby move. Goal: 10 kicks in 2 hours.
            </p>
            <KickCounter />
          </Card>
        </div>

        {/* Right Column */}
        <div className="space-y-4">
          {/* Movement Heatmap */}
          <Card>
            <SectionHeader title="24-hour movement pattern" />
            <p className="text-[12px] text-muted mb-3">
              Hourly kick counts · each dot = kicks per hour
            </p>
            <div
              className="grid gap-2"
              style={{ gridTemplateColumns: "repeat(8,1fr)" }}
            >
              {hourlyKicks.map((count, idx) => (
                <div
                  key={idx}
                  className="aspect-square rounded-full flex items-center justify-center text-[9px] font-bold cursor-pointer transition-transform hover:scale-110"
                  style={{
                    background:
                      count >= 18
                        ? "#E05675"
                        : count >= 12
                        ? "#D0C2F0"
                        : "#EAE4F5",
                    color:
                      count >= 18
                        ? "white"
                        : count >= 12
                        ? "#5234A0"
                        : "#8E87A6",
                  }}
                >
                  {count}
                </div>
              ))}
            </div>
            <div className="flex gap-4 mt-3 text-[11px] text-muted">
              {[
                ["#E05675", "Active"],
                ["#D0C2F0", "Moderate"],
                ["#EAE4F5", "Low"],
              ].map(([color, label]) => (
                <span key={label} className="flex items-center gap-1.5">
                  <span
                    className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{ background: color }}
                  />
                  <span>{label}</span>
                </span>
              ))}
            </div>
          </Card>

          {/* Fetal AI Analysis */}
          <HighlightCard>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 rounded-full bg-plum text-white flex items-center justify-center text-sm">
                ✨
              </div>
              <div>
                <div className="font-display text-[14px] font-semibold text-plum-dark">
                  Fetal AI Analysis (Demo)
                </div>
                <div className="text-[10px] text-muted">Updated just now</div>
              </div>
            </div>
            <p className="text-[13px] text-mid leading-relaxed">
              Baby&apos;s HR of <strong>{fhrLive} bpm</strong> is within the healthy
              110–160 bpm range. CNN hypoxia score of{" "}
              {data.fetal.cnn_raw.hypoxia_probability} indicates low risk.{" "}
              {data.fetal.movement_count} kicks/hour is reassuring.
            </p>
            <div className="mt-3 p-2.5 bg-white/70 rounded-xl text-[11px] text-muted">
              If kicks drop below 10 in 2 hours during active periods, contact
              your midwife immediately.
            </div>
            <ApiResponseViewer
              data={{
                live_sensor_hardware: {
                  piezo_acoustic: sensorData.piezo,
                  accel_x: sensorData.accel_x,
                  accel_y: sensorData.accel_y,
                  accel_z: sensorData.accel_z,
                  status: status,
                },
                simulated_fetal_data: data.fetal,
              }}
            />
          </HighlightCard>
        </div>
      </div>
    </div>
  );
}
