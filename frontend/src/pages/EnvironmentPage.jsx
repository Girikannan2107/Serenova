import React from "react";
import {
  SectionHeader,
  Card,
  StatusBadge,
} from "../components/common/CommonComponents";
import { environmentalData } from "../data/mockData";

export default function EnvironmentPage() {
  const cards = [
    {
      ...environmentalData.aqi,
      bg: "#FDF5DC",
      border: "#e8c96a",
      col: "#854F0B",
    },
    {
      ...environmentalData.uv,
      bg: "#FFF0F4",
      border: "#FFD6E0",
      col: "#C03D5A",
    },
    {
      ...environmentalData.heat,
      bg: "#E6F7EE",
      border: "#a3d9bc",
      col: "#1D7A50",
    },
  ];

  const activities = [
    {
      e: "🏊",
      t: "Indoor swimming — excellent",
      d: "Cool water, low impact, full body. 30 min recommended.",
      ok: true,
    },
    {
      e: "🧘",
      t: "Prenatal yoga — excellent",
      d: "Gentle stretching, breathing. Safe in AC indoors.",
      ok: true,
    },
    {
      e: "🚶",
      t: "Outdoor walk — caution",
      d: "Under 15 min, before 8 AM or after 6 PM only.",
      ok: null,
    },
    {
      e: "🏃",
      t: "Vigorous outdoor exercise — avoid",
      d: "AQI + heat index combination is risky. Skip today.",
      ok: false,
    },
  ];

  const bgMap = {
    true: "bg-sage-light",
    null: "bg-amber-light",
    false: "bg-red-50",
  };
  const textMap = {
    true: "text-sage-dark",
    null: "text-amber-dark",
    false: "text-red-700",
  };

  return (
    <div className="space-y-5 animate-fade-up">
      {/* 3 Environmental Metric Cards */}
      <div className="grid gap-4 sm:grid-cols-3">
        {cards.map((c, i) => (
          <div
            key={i}
            className="rounded-2xl p-5"
            style={{
              background: c.bg,
              border: `1px solid ${c.border}`,
            }}
          >
            <div className="text-2xl mb-2">{c.icon}</div>
            <div
              className="font-display text-5xl font-bold leading-none"
              style={{ color: c.col }}
            >
              {c.value}
            </div>
            <div
              className="text-[13px] font-semibold mt-2"
              style={{ color: c.col }}
            >
              {c.label}
            </div>
            <div
              className="text-[11px] mt-0.5"
              style={{ color: c.col, opacity: 0.7 }}
            >
              {c.sub}
            </div>
            <div
              className="text-[12px] mt-3 pt-3 border-t leading-relaxed"
              style={{ borderColor: `${c.border}80`, color: c.col }}
            >
              {c.advice}
            </div>
          </div>
        ))}
      </div>

      {/* Safe Activities List */}
      <Card>
        <SectionHeader
          title="Safe activities today"
          right={<StatusBadge status="watch" label="Moderate caution" />}
        />
        {activities.map((act, i) => (
          <div
            key={i}
            className={`flex gap-3 p-3 rounded-xl mb-2 ${bgMap[String(act.ok)]}`}
          >
            <span className="text-xl">{act.e}</span>
            <div>
              <div className={`text-[13px] font-semibold ${textMap[String(act.ok)]}`}>
                {act.t}
              </div>
              <div className="text-[11px] text-muted mt-0.5">{act.d}</div>
            </div>
          </div>
        ))}
      </Card>
    </div>
  );
}
