import React, { useState } from "react";
import {
  StatusBadge,
  SectionHeader,
  Card,
  HighlightCard,
  ApiResponseViewer,
} from "../components/common/CommonComponents";

export default function DoctorPortalPage({ data, sensorState }) {
  const [note, setNote] = useState("");
  const [sentToast, setSentToast] = useState(false);

  const metrics = data.maternal.metrics;
  const risk = data.risk;
  const sensorData = sensorState?.sensorData || {};
  const status = sensorState?.status || {};
  const connStatus = sensorState?.connectionStatus || "disconnected";

  const handleSendNote = () => {
    if (note.trim()) {
      setSentToast(true);
      setTimeout(() => setSentToast(false), 3000);
      setNote("");
    }
  };

  const clinicalVitals = [
    { p: "Heart rate (baseline)", v: `${metrics.heart_rate.value} bpm`, s: metrics.heart_rate.status },
    {
      p: "Blood pressure",
      v: `${metrics.blood_pressure.systolic}/${metrics.blood_pressure.diastolic} mmHg`,
      s: metrics.blood_pressure.status,
    },
    { p: "HRV", v: `${metrics.hrv.value} ms`, s: metrics.hrv.status },
    { p: "SpO₂", v: `${metrics.spo2.value}%`, s: metrics.spo2.status },
    { p: "Glucose", v: `${metrics.glucose.value} mg/dL`, s: metrics.glucose.status },
    {
      p: "Skin Temperature",
      v: sensorData.temperature_c != null ? `${sensorData.temperature_c}°C (Live)` : `${metrics.temperature.value}°C (Demo)`,
      s: "normal",
    },
    { p: "Weight", v: `${metrics.weight.value} kg`, s: metrics.weight.status },
    { p: "Respiratory rate", v: `${metrics.respiratory_rate.value} rpm`, s: metrics.respiratory_rate.status },
    { p: "Fetal HR (baseline)", v: `${data.fetal.fhr} bpm`, s: data.fetal.fetal_status },
    { p: "CNN hypoxia", v: `${data.fetal.cnn_raw.hypoxia_probability}`, s: "normal" },
    { p: "Risk score", v: `${risk.risk_score}/100`, s: risk.risk_level },
  ];

  return (
    <div className="space-y-5 animate-fade-up">
      {/* Patient Header Banner */}
      <div
        className="rounded-2xl p-5"
        style={{ background: "#16121E", color: "white" }}
      >
        <div className="flex items-start justify-between">
          <div>
            <div className="font-display text-xl font-semibold text-white">
              Priya Sharma
            </div>
            <div className="text-[12px] opacity-60 mt-1">
              28 weeks · G1P0 · Due: Jul 12, 2025 · DoB: Mar 14, 1996
            </div>
            <div
              className="mt-2 inline-block px-3 py-1 rounded-full text-[12px] font-semibold"
              style={{ background: "rgba(255,255,255,.12)" }}
            >
              Risk score: {risk.risk_score}/100 · {risk.risk_level} risk
            </div>
          </div>
          <button
            className="px-3.5 py-1.5 rounded-xl text-[12px] font-semibold border text-white hover:bg-white/10 transition-colors"
            style={{ borderColor: "rgba(255,255,255,.2)" }}
          >
            📄 Export PDF
          </button>
        </div>
      </div>

      {/* Grid: Vitals & Notes */}
      <div className="grid gap-5 lg:grid-cols-2">
        {/* Full Vitals Table */}
        <Card>
          <SectionHeader
            title="Full clinical vitals"
            right={<StatusBadge status="watch" label="1 item watching" />}
          />
          {clinicalVitals.map((v, idx) => (
            <div
              key={idx}
              className={`flex items-center py-2.5 ${
                idx > 0 ? "border-t border-border" : ""
              }`}
            >
              <span className="text-[13px] text-mid flex-1">{v.p}</span>
              <span className="text-[13px] font-bold text-ink mr-3">{v.v}</span>
              <StatusBadge status={v.s} />
            </div>
          ))}
        </Card>

        {/* Risk Analysis & Doctor Notes */}
        <div className="space-y-4">
          <HighlightCard>
            <div className="flex items-center gap-2 mb-3">
              <span className="text-xl">✨</span>
              <div className="font-display text-[14px] font-semibold text-plum-dark">
                Risk factor analysis
              </div>
            </div>
            {risk.positives.map((pos, idx) => (
              <div
                key={idx}
                className="flex gap-2 text-[12px] text-ink py-1.5 border-b border-white/50"
              >
                <span className="text-sage font-bold flex-shrink-0">✓</span>
                {pos}
              </div>
            ))}
            {risk.watchful_items.map((watch, idx) => (
              <div
                key={idx}
                className="flex gap-2 text-[12px] text-ink py-1.5 border-b border-white/30"
              >
                <span className="text-amber font-bold flex-shrink-0">~</span>
                {watch}
              </div>
            ))}
            {risk.rag && (
              <div className="mt-3 p-2.5 bg-white/60 rounded-xl text-[11px] text-mid space-y-1">
                <div>
                  <strong>RAG summary:</strong> {risk.rag.summary}
                </div>
                <div>
                  <strong>Decision:</strong> {risk.rag.decision}
                </div>
                <div>
                  <strong>Explanation:</strong> {risk.rag.explanation}
                </div>
              </div>
            )}
            <div className="mt-2 p-2.5 bg-white/50 rounded-xl text-[11px] text-muted italic">
              {risk.doctor_summary}
            </div>
          </HighlightCard>

          <Card>
            <SectionHeader title="Add clinical note" />
            <textarea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              rows={3}
              className="w-full px-3.5 py-2.5 border border-border rounded-xl text-[13px] font-body outline-none resize-none text-ink focus:border-plum transition-colors"
              placeholder="Type a clinical note for this patient…"
            />
            <div className="flex gap-2 mt-3">
              <button
                onClick={handleSendNote}
                className="flex-1 py-2.5 rounded-full bg-ink text-white text-[13px] font-semibold hover:opacity-90 transition-opacity"
              >
                Send to patient
              </button>
              <button
                onClick={() => setNote("")}
                className="px-4 rounded-full bg-rose-light text-rose text-[13px] font-semibold hover:bg-rose-medium transition-colors"
              >
                Clear
              </button>
            </div>
            {sentToast && (
              <div className="mt-2 p-2.5 bg-sage-light rounded-xl text-[12px] text-sage-dark font-semibold animate-fade-up">
                ✓ Note sent to patient&apos;s Serenova app
              </div>
            )}
          </Card>
        </div>
      </div>

      <ApiResponseViewer data={data} />
    </div>
  );
}
