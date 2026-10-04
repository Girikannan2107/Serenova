import React, { useState, useEffect } from "react";
import FetusModel3D from "../components/3d/FetusModel3D";
import {
  StatusBadge,
  ProgressBar,
  MetricCard,
  SectionHeader,
  AlertCard,
  Card,
  HighlightCard,
  PulsingDot,
  ApiResponseViewer,
} from "../components/common/CommonComponents";
import { fetalGrowthData } from "../data/fetalGrowthData";

export default function DashboardPage({ data, week, setPage, sensorState }) {
  const currentGrowth = fetalGrowthData[week] || fetalGrowthData[28];
  const maternalMetrics = data.maternal.metrics;
  const riskInfo = data.risk;
  const [fhrLive, setFhrLive] = useState(data.fetal.fhr);

  // Live sensor telemetry values
  const sensorData = sensorState?.sensorData || {};
  const status = sensorState?.status || {};
  const connStatus = sensorState?.connectionStatus || "disconnected";
  const isConnected = connStatus === "connected";

  useEffect(() => {
    const timer = setInterval(() => {
      setFhrLive((prev) =>
        Math.max(118, Math.min(162, prev + Math.round((Math.random() - 0.5) * 5)))
      );
    }, 3000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="space-y-5 animate-fade-up">
      {/* Real vs Mock Architectural Distinction Banner */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 bg-white border border-border rounded-2xl shadow-card">
        <div className="flex items-center gap-2 text-[12px]">
          <span className="font-bold text-ink uppercase tracking-wider text-[11px]">
            Live Sensor Data:
          </span>
          <span
            className={`inline-flex items-center gap-1.5 font-semibold px-2.5 py-0.5 rounded-full border ${
              connStatus === "connected"
                ? "bg-sage-light text-sage-dark border-sage/30"
                : connStatus === "stale"
                ? "bg-amber-light text-amber-dark border-amber/30"
                : "bg-[#F5F0FF] text-[#7052B8] border-[#EAE4F5]"
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
            {connStatus.toUpperCase()} · {status.device_id || "serenova-pico-01"}
          </span>
          {isConnected && (
            <span className="text-[11px] text-muted hidden sm:inline">
              (Seq: #{sensorState?.sequence || 0} · Loss: {sensorState?.packetLoss || 0})
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 text-[12px]">
          <span className="font-bold text-muted uppercase tracking-wider text-[11px]">
            Clinical Assessment:
          </span>
          <span className="inline-flex items-center gap-1 font-semibold px-2.5 py-0.5 rounded-full bg-[#FDF2F8] text-[#BE185D] border border-[#FBCFE8]">
            ● DEMO / SIMULATED
          </span>
        </div>
      </div>

      {/* 3D Growth Banner */}
      <div
        className="rounded-3xl overflow-hidden"
        style={{
          background: "linear-gradient(135deg,#FFF0F4,#F0EBFF)",
          border: "1px solid #FFD6E0",
        }}
      >
        <div className="flex gap-0 min-h-0">
          <div className="flex-1 p-5 min-w-0">
            <div className="text-[11px] font-semibold text-rose uppercase tracking-widest mb-2">
              Week {week} · {currentGrowth.trimester === 3 ? "Third" : "Second"}{" "}
              Trimester
            </div>
            <div className="font-display text-xl font-bold text-ink mb-1">
              Baby is the size of a{" "}
              <span className="text-rose italic">
                {currentGrowth.name} {currentGrowth.emoji}
              </span>
            </div>
            <div className="text-[13px] text-mid mb-3">
              {currentGrowth.size} · {currentGrowth.weight}
            </div>
            <div className="space-y-1.5 mb-3">
              {currentGrowth.facts.map((fact, idx) => (
                <div key={idx} className="flex items-center gap-2 text-[12px] text-mid">
                  <div className="w-1.5 h-1.5 rounded-full bg-rose flex-shrink-0" />
                  {fact}
                </div>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <PulsingDot color="rose" />
              <span className="text-[12px] font-semibold text-rose">
                {fhrLive} bpm live fetal HR (demo)
              </span>
            </div>
          </div>
          <div className="w-44 h-48 flex-shrink-0 relative">
            <FetusModel3D week={week} />
            <div className="absolute bottom-2 inset-x-0 text-center text-[9px] text-muted">
              3D · Gestational week {week}
            </div>
          </div>
        </div>
      </div>

      {/* REAL HARDWARE SENSOR TELEMETRY CARDS */}
      <Card>
        <SectionHeader
          title="Hardware Telemetry (Live Pico W Sensors)"
          right={
            <StatusBadge
              status={
                connStatus === "connected"
                  ? "normal"
                  : connStatus === "stale"
                  ? "watch"
                  : "alert"
              }
              label={
                connStatus === "connected"
                  ? "Streaming"
                  : connStatus === "stale"
                  ? "Stale"
                  : "No Signal"
              }
            />
          }
        />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MetricCard
            icon="⚡"
            value={sensorData?.ecg != null ? sensorData.ecg : "—"}
            unit="ADC"
            label="Raw ECG Signal"
            status={sensorData?.ecg != null ? "normal" : "watch"}
            sub="AD8232 Lead I"
          />
          <MetricCard
            icon="💡"
            value={sensorData?.ppg_ir != null ? sensorData.ppg_ir : "—"}
            unit="cnt"
            label="PPG IR Pulse"
            status={sensorData?.ppg_ir != null ? "normal" : "watch"}
            sub={sensorData?.ppg_red != null ? `Red: ${sensorData.ppg_red}` : "MAX30102"}
          />
          <MetricCard
            icon="🌡️"
            value={
              sensorData?.temperature_c != null
                ? sensorData.temperature_c
                : maternalMetrics.temperature.value
            }
            unit="°C"
            label="Skin Temperature"
            status={sensorData?.temperature_c != null ? "normal" : "watch"}
            sub={sensorData?.temperature_c != null ? "Live DS18B20" : "Simulated"}
          />
          <MetricCard
            icon="🔊"
            value={sensorData?.piezo != null ? sensorData.piezo : "—"}
            unit="raw"
            label="Piezo / Acoustic"
            status={sensorData?.piezo != null ? "normal" : "watch"}
            sub="Abdominal sensor"
          />
        </div>

        {/* IMU & Transmission details */}
        <div className="mt-3 p-3 bg-[#F5F0FF] rounded-xl flex flex-wrap items-center justify-between gap-2 text-[11px] text-mid">
          <div className="flex items-center gap-3">
            <span>
              <strong>Accel (X/Y/Z):</strong>{" "}
              {sensorData?.accel_x ?? "—"} / {sensorData?.accel_y ?? "—"} /{" "}
              {sensorData?.accel_z ?? "—"} g
            </span>
            <span className="hidden sm:inline">
              <strong>Gyro (X/Y/Z):</strong> {sensorData?.gyro_x ?? "—"} /{" "}
              {sensorData?.gyro_y ?? "—"} / {sensorData?.gyro_z ?? "—"}
            </span>
          </div>
          <div className="text-muted">
            Packets: {sensorState?.packetCount ?? 0} · Loss:{" "}
            {sensorState?.packetLoss ?? 0}
          </div>
        </div>
      </Card>

      {/* Vitals Grid - Row 1 (Clinical Mock Overview) */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard
          icon="❤️"
          value={maternalMetrics.heart_rate.value}
          unit="bpm"
          label="Heart rate (maternal)"
          status={maternalMetrics.heart_rate.status}
          sub="Clinical baseline"
        />
        <MetricCard
          icon="🩸"
          value={`${maternalMetrics.blood_pressure.systolic}/${maternalMetrics.blood_pressure.diastolic}`}
          unit="mmHg"
          label="Blood pressure"
          status={maternalMetrics.blood_pressure.status}
          sub="Clinical baseline"
        />
        <MetricCard
          icon="💚"
          value={maternalMetrics.hrv.value}
          unit="ms"
          label="HRV"
          status={maternalMetrics.hrv.status}
          trend="↘ drift"
          sub="Clinical baseline"
        />
        <MetricCard
          icon="🫁"
          value={maternalMetrics.spo2.value}
          unit="%"
          label="SpO₂"
          status={maternalMetrics.spo2.status}
          sub="Clinical baseline"
        />
      </div>

      {/* Vitals Grid - Row 2 */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard
          icon="🔬"
          value={maternalMetrics.glucose.value}
          unit="mg/dL"
          label="Glucose"
          status={maternalMetrics.glucose.status}
          sub="Clinical baseline"
        />
        <MetricCard
          icon="⚖️"
          value={maternalMetrics.weight.value}
          unit="kg"
          label="Weight"
          status={maternalMetrics.weight.status}
          sub="Clinical baseline"
        />
        <MetricCard
          icon="👶"
          value={fhrLive}
          unit="bpm"
          label="Fetal HR (live)"
          status="normal"
          sub="CTG NST monitor"
        />
        <MetricCard
          icon="🫀"
          value={maternalMetrics.respiratory_rate.value}
          unit="rpm"
          label="Respiratory rate"
          status={maternalMetrics.respiratory_rate.status}
          sub="Clinical baseline"
        />
      </div>

      {/* Alerts and Progress Section */}
      <div className="grid gap-5 lg:grid-cols-2">
        <div>
          <SectionHeader
            title="Alerts & Notifications"
            right={
              <button
                onClick={() => setPage("timeline")}
                className="text-[12px] text-rose font-semibold hover:underline"
              >
                View all
              </button>
            }
          />
          <AlertCard
            type="calm"
            title="Blood pressure stable — excellent"
            body="BP within normal range for 21 consecutive days. Consistent and healthy readings."
            time="2 hours ago"
          />
          <AlertCard
            type="watch"
            title="HRV declining trend detected"
            body="HRV at 66 ms (from 78 ms). Drift anomaly logged. Worth mentioning at next appointment."
            time="Today, 8:14 AM"
          />
          <AlertCard
            type="calm"
            title="Fetal kick activity: excellent"
            body="20 kicks in last hour — strong movement consistent with 28-week development."
            time="Today, 9:30 AM"
          />
        </div>

        <div>
          <SectionHeader
            title="Weekly progress"
            right={
              <StatusBadge
                status={riskInfo.risk_level}
                label={`${riskInfo.risk_score}/100 · ${riskInfo.risk_level} risk`}
              />
            }
          />
          <Card>
            {[
              { label: "Pregnancy journey", pct: 67, color: "#E05675" },
              { label: "Sleep quality", pct: 82, color: "#7052B8" },
              { label: "Nutrition score", pct: 75, color: "#2A9968" },
              { label: "Activity goal", pct: 48, color: "#C48A08" },
            ].map((item) => (
              <div key={item.label} className="mb-3 last:mb-0">
                <div className="flex justify-between text-[12px] mb-1">
                  <span className="text-mid">{item.label}</span>
                  <span className="font-semibold" style={{ color: item.color }}>
                    {item.pct}%
                  </span>
                </div>
                <ProgressBar value={item.pct} color={item.color} />
              </div>
            ))}
            <div className="flex gap-2 mt-4 pt-4 border-t border-border">
              <button
                onClick={() => setPage("ai")}
                className="flex-1 py-2 rounded-full bg-rose text-white text-[12px] font-semibold hover:bg-rose-dark transition-colors"
              >
                Ask AI ✨
              </button>
              <button
                onClick={() => setPage("doctor")}
                className="flex-1 py-2 rounded-full bg-rose-light text-rose text-[12px] font-semibold hover:bg-rose-medium transition-colors"
              >
                Doctor report
              </button>
            </div>
          </Card>
        </div>
      </div>

      {/* AI Risk Analysis Highlight Card */}
      <HighlightCard>
        <div className="flex items-start gap-3 mb-4">
          <div className="w-9 h-9 rounded-full bg-plum flex items-center justify-center text-white text-base flex-shrink-0">
            ✨
          </div>
          <div className="flex-1">
            <div className="font-display text-[15px] font-semibold text-plum-dark">
              AI Risk Analysis (Demo) · Week {week}
            </div>
            <div className="text-[11px] text-muted mt-0.5">
              CNN hypoxia: {data.fetal.cnn_raw.hypoxia_probability} ·{" "}
              {riskInfo.risk_level} risk · {riskInfo.risk_score}/100
            </div>
          </div>
          <StatusBadge
            status={riskInfo.risk_level}
            label={`${
              riskInfo.risk_level.charAt(0).toUpperCase() +
              riskInfo.risk_level.slice(1)
            } risk`}
          />
        </div>

        <p className="text-[13px] text-mid leading-relaxed mb-3">
          {riskInfo.plain_language_summary}
        </p>

        <div className="space-y-1.5">
          {riskInfo.positives.map((item, idx) => (
            <div key={idx} className="flex gap-2 text-[12px] text-ink">
              <span className="text-sage font-bold flex-shrink-0">✓</span>
              {item}
            </div>
          ))}
          {riskInfo.watchful_items.map((item, idx) => (
            <div key={idx} className="flex gap-2 text-[12px] text-ink">
              <span className="text-amber font-bold flex-shrink-0">~</span>
              {item}
            </div>
          ))}
        </div>

        {data.maternal.drift_anomalies?.length > 0 && (
          <div className="mt-3 p-2.5 bg-amber-light border border-amber/20 rounded-xl text-[12px] text-amber-dark">
            ⚠ Drift anomaly: {data.maternal.drift_anomalies[0]}
          </div>
        )}

        <div className="mt-3 p-2.5 bg-white/70 rounded-xl text-[11px] text-muted italic">
          {riskInfo.doctor_summary}
        </div>

        <ApiResponseViewer
          data={{
            live_hardware_sensor: {
              status: sensorState?.status,
              sensors: sensorState?.sensorData,
            },
            simulated_clinical_data: {
              fetal: data.fetal,
              maternal: data.maternal,
              risk: data.risk,
            },
          }}
        />
      </HighlightCard>

      {/* Condition Monitoring Grid */}
      <Card>
        <SectionHeader title="Condition monitoring" />
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
          {riskInfo.conditions.map((cond, idx) => (
            <div
              key={idx}
              className="p-3 rounded-xl border border-border text-center"
            >
              <div className="text-[11px] text-muted mb-1.5 leading-tight">
                {cond.name}
              </div>
              <StatusBadge status={cond.status} />
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
