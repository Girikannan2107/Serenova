import React from "react";
import { PulsingDot } from "../common/CommonComponents";

export default function TopHeader({
  title,
  week,
  mode,
  riskScore,
  riskLevel,
  onOpenMobileMenu,
  onOpenAi,
  sensorState,
}) {
  const connStatus = sensorState?.connectionStatus || "disconnected";
  const backendAvailable = sensorState?.backendAvailable ?? true;
  const isWs = sensorState?.isWebSocketActive;

  const sensorBadgeConfig = !backendAvailable
    ? {
        label: "Sensor Backend Offline",
        color: "amber",
        bg: "#FFF5EB",
        textColor: "#B45309",
        borderColor: "rgba(217,119,6,.2)",
      }
    : connStatus === "connected"
    ? {
        label: `LIVE SENSOR · Connected${isWs ? " (WS)" : " (HTTP)"}`,
        color: "sage",
        bg: "#E6F7EE",
        textColor: "#1D7A50",
        borderColor: "rgba(42,153,104,.25)",
      }
    : connStatus === "stale"
    ? {
        label: "LIVE SENSOR · Stale (2-5s)",
        color: "amber",
        bg: "#FEF3C7",
        textColor: "#92400E",
        borderColor: "rgba(245,158,11,.3)",
      }
    : {
        label: "LIVE SENSOR · Disconnected",
        color: "rose",
        bg: "#F5F0FF",
        textColor: "#7052B8",
        borderColor: "#EAE4F5",
      };

  return (
    <header className="h-14 bg-white border-b border-border flex items-center px-5 gap-3 flex-shrink-0">
      <button
        onClick={onOpenMobileMenu}
        className="md:hidden w-8 h-8 rounded-lg bg-[#F5F0FF] flex items-center justify-center text-mid"
      >
        ☰
      </button>

      <div className="flex items-center gap-2 min-w-0">
        <span className="font-display text-[17px] font-bold text-ink truncate">
          {title}
        </span>
        <span className="text-[12px] text-muted hidden sm:inline">
          · Week {week} · {mode === "postpartum" ? "Postpartum" : "Pregnancy"}
        </span>
      </div>

      <div className="ml-auto flex items-center gap-2">
        {/* Real Sensor Telemetry Status Indicator */}
        <div
          className="hidden md:flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-1 rounded-full border transition-all"
          style={{
            background: sensorBadgeConfig.bg,
            color: sensorBadgeConfig.textColor,
            borderColor: sensorBadgeConfig.borderColor,
          }}
          title={
            backendAvailable
              ? `Pico W status: ${connStatus} | Device: ${sensorState?.status?.device_id || "N/A"} | Seq: ${sensorState?.sequence || 0}`
              : "FastAPI backend is unreachable on configured API base URL"
          }
        >
          <PulsingDot color={sensorBadgeConfig.color} />
          <span>{sensorBadgeConfig.label}</span>
        </div>

        {/* Mock Clinical Data Indication */}
        <div
          onClick={onOpenAi}
          className="hidden sm:flex items-center gap-1.5 text-[12px] font-semibold cursor-pointer px-3 py-1.5 rounded-full border"
          style={{
            background: "#FDF2F8",
            color: "#BE185D",
            borderColor: "rgba(219,39,119,.25)",
          }}
          title="Clinical assessment scores and recommendations are simulated mock demonstration data"
        >
          <span className="text-[10px] px-1.5 py-0.2 rounded bg-pink-100 text-pink-700 uppercase tracking-wider font-bold">
            DEMO
          </span>
          Risk: {riskScore}/100 · {riskLevel}
        </div>

        <button
          className="w-8 h-8 rounded-lg flex items-center justify-center text-[15px] hover:bg-rose-light transition-colors"
          style={{ background: "#F5F0FF" }}
          title="Notifications"
        >
          🔔
        </button>
      </div>
    </header>
  );
}
