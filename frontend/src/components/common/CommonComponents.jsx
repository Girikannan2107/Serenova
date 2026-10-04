import React from "react";

export function StatusBadge({ status, label }) {
  const styles = {
    normal: "bg-sage-light text-sage-dark border border-sage/30",
    watch: "bg-amber-light text-amber-dark border border-amber/30",
    alert: "bg-red-50 text-red-700 border border-red-300",
    low: "bg-plum-light text-plum-dark border border-plum/30",
    medium: "bg-amber-light text-amber-dark border border-amber/30",
    high: "bg-red-50 text-red-700 border border-red-300",
  };
  const currentStyle = styles[status] || styles.low;
  return (
    <span className={`inline-block text-[11px] font-semibold px-2.5 py-0.5 rounded-full ${currentStyle}`}>
      {label || (status ? status.charAt(0).toUpperCase() + status.slice(1) : "—")}
    </span>
  );
}

export function ProgressBar({ value, max = 100, color = "#E05675", className = "" }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className={`h-1.5 rounded-full bg-border overflow-hidden ${className}`}>
      <div
        className="h-full rounded-full transition-all duration-700 ease-out"
        style={{ width: `${pct}%`, background: color }}
      />
    </div>
  );
}

export function MetricCard({ icon, value, unit, label, status, sub, onClick, trend }) {
  const colorMap = {
    normal: "#2A9968",
    watch: "#C48A08",
    alert: "#D03030",
    low: "#7052B8",
  };
  const iconStatusMap = {
    normal: "✓",
    watch: "~",
    alert: "!",
    low: "↓",
  };
  const statusColor = colorMap[status] || "#8E87A6";
  const statusIcon = iconStatusMap[status] || "·";

  return (
    <div
      onClick={onClick}
      className="group bg-white border border-border rounded-2xl p-4 shadow-card cursor-pointer hover:shadow-float hover:-translate-y-1 transition-all duration-200"
    >
      <div className="text-xl mb-2">{icon}</div>
      <div className="font-display text-[22px] font-bold leading-none text-ink">
        {value}
        {unit && (
          <span className="text-sm font-body font-normal text-muted ml-1">
            {unit}
          </span>
        )}
      </div>
      <div className="text-xs text-muted mt-1.5">{label}</div>
      {sub && <div className="text-[10px] text-muted mt-0.5 opacity-70">{sub}</div>}
      <div
        className="text-[11px] font-semibold mt-2.5 flex items-center gap-1"
        style={{ color: statusColor }}
      >
        <span>{statusIcon}</span>
        <span className="capitalize">
          {status === "normal" ? "Normal" : status === "watch" ? "Monitor" : status}
        </span>
        {trend && (
          <span className="ml-auto text-muted font-normal text-[10px]">
            {trend}
          </span>
        )}
      </div>
    </div>
  );
}

export function SectionHeader({ title, right, className = "" }) {
  return (
    <div className={`flex items-center justify-between mb-4 ${className}`}>
      <h3 className="font-display text-base font-semibold text-ink">{title}</h3>
      {right}
    </div>
  );
}

export function AlertCard({ type = "calm", title, body, time }) {
  const styles = {
    calm: { wrap: "bg-sage-light border-sage/25", dot: "bg-sage", t: "text-ink" },
    watch: { wrap: "bg-amber-light border-amber/25", dot: "bg-amber", t: "text-ink" },
    alert: { wrap: "bg-red-50 border-red-300", dot: "bg-red-500", t: "text-ink" },
  };
  const s = styles[type] || styles.calm;
  return (
    <div className={`${s.wrap} border rounded-xl p-3.5 flex gap-3 mb-3`}>
      <div className={`${s.dot} w-2 h-2 rounded-full flex-shrink-0 mt-1.5`} />
      <div className="flex-1 min-w-0">
        <div className={`text-[13px] font-semibold ${s.t}`}>{title}</div>
        <div className="text-[12px] text-mid leading-relaxed mt-0.5">{body}</div>
        {time && <div className="text-[11px] text-muted mt-1">{time}</div>}
      </div>
    </div>
  );
}

export function Card({ children, className = "", padding = true }) {
  return (
    <div
      className={`bg-white border border-border rounded-2xl shadow-card ${
        padding ? "p-5" : ""
      } ${className}`}
    >
      {children}
    </div>
  );
}

export function HighlightCard({ children, className = "" }) {
  return (
    <div
      className={`rounded-2xl p-5 ${className}`}
      style={{
        background: "linear-gradient(135deg,#F0EBFF,#FFF0F4)",
        border: "1.5px solid #D0C2F0",
      }}
    >
      {children}
    </div>
  );
}

export function PulsingDot({ color = "rose" }) {
  const colors = {
    rose: "bg-rose",
    sage: "bg-sage",
    amber: "bg-amber",
  };
  return (
    <span
      className={`inline-block w-1.5 h-1.5 rounded-full ${
        colors[color] || "bg-rose"
      } animate-blink`}
    />
  );
}

export function HeartRateWave() {
  const points = [];
  for (let x = 0; x <= 300; x += 8) {
    let y = 32;
    if (x % 80 < 4) y = 8;
    else if (x % 80 < 8) y = 52;
    else if (x % 80 < 12) y = 16;
    else if (x % 80 < 16) y = 32;
    points.push(`${x},${y}`);
  }
  const pts = points.join(" ");
  return (
    <svg viewBox="0 0 300 64" className="w-full h-12">
      <polyline
        points={pts}
        fill="none"
        stroke="#FFD6E0"
        strokeWidth="5"
        strokeLinecap="round"
        opacity=".5"
      />
      <polyline
        points={pts}
        fill="none"
        stroke="#E05675"
        strokeWidth="1.8"
        strokeLinecap="round"
        className="wave-path"
      />
    </svg>
  );
}

export function ApiResponseViewer({ data }) {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="mt-4">
      <button
        onClick={() => setOpen((prev) => !prev)}
        className="flex items-center gap-2 text-[11px] font-semibold text-plum hover:text-plum-dark transition-colors"
      >
        <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
          <rect x="1" y="2" width="11" height="9" rx="2" stroke="currentColor" strokeWidth="1.2" />
          <path d="M3 5h7M3 7.5h5" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
        </svg>
        {open ? "Hide" : "View"} backend API response
      </button>
      {open && (
        <pre className="mt-2 p-3 bg-ink rounded-xl text-green-400 text-[10px] leading-relaxed overflow-auto max-h-60">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}

export function TypingIndicator() {
  return (
    <div className="flex gap-1 items-center px-3.5 py-2.5 bg-white border border-border rounded-2xl rounded-bl-sm w-fit">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-muted inline-block animate-bounce3"
          style={{ animationDelay: `${i * 0.2}s` }}
        />
      ))}
    </div>
  );
}

export function MoodSelector({ value, onChange }) {
  const moods = [
    { key: "great", emoji: "😊", label: "Great", color: "sage" },
    { key: "okay", emoji: "😐", label: "Okay", color: "amber" },
    { key: "low", emoji: "😔", label: "Low", color: "rose" },
    { key: "anxious", emoji: "😰", label: "Anxious", color: "plum" },
    { key: "tired", emoji: "😴", label: "Tired", color: "mid" },
  ];
  const descriptions = {
    great: "Great to hear you're feeling positive today!",
    okay: "Thanks for checking in. Some days are just okay — rest is valid.",
    low: "Feeling low is real. If this persists, please reach out to someone.",
    anxious: "Post-delivery anxiety is very common. Support resources are below.",
    tired: "Rest is healing. Sleep when baby sleeps today.",
  };
  const colorBorders = {
    sage: "border-sage",
    amber: "border-amber",
    rose: "border-rose",
    plum: "border-plum",
    mid: "border-mid",
  };

  return (
    <div>
      <div className="flex gap-2 mb-3">
        {moods.map((m) => (
          <button
            key={m.key}
            onClick={() => onChange(m.key)}
            className={`flex-1 py-3 px-1 rounded-xl border-2 transition-all text-center
              ${
                value === m.key
                  ? `${colorBorders[m.color]} scale-105 bg-opacity-10`
                  : "border-border bg-white hover:border-border/60"
              }`}
          >
            <span className="block text-lg">{m.emoji}</span>
            <span className="block text-[10px] font-semibold text-muted mt-1">
              {m.label}
            </span>
          </button>
        ))}
      </div>
      {value && (
        <div className="p-3 bg-rose-light rounded-xl text-[12px] text-mid animate-fade-up">
          {descriptions[value]}
        </div>
      )}
    </div>
  );
}
