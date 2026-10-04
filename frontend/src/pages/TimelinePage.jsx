import React from "react";
import { SectionHeader, Card } from "../components/common/CommonComponents";

export default function TimelinePage() {
  const activities = [
    {
      c: "#2A9968",
      t: "Blood pressure logged",
      b: "119/74 mmHg · stable for 21 days.",
      tm: "Today, 9:47 AM",
    },
    {
      c: "#E05675",
      t: "Fetal HR session complete",
      b: "137 bpm · CNN hypoxia 0.18 · 20 min.",
      tm: "Today, 9:30 AM",
    },
    {
      c: "#C48A08",
      t: "HRV drift anomaly flagged",
      b: "5-day decline: 78→66 ms. Logged.",
      tm: "Today, 8:14 AM",
    },
    {
      c: "#7052B8",
      t: "Morning vitals synced",
      b: "Apple Watch data · Sleep 82/100 · HR 84.",
      tm: "Today, 7:00 AM",
    },
    {
      c: "#2A9968",
      t: "Iron goal met",
      b: "27 mg target hit — third consecutive day.",
      tm: "Yesterday, 8:30 PM",
    },
    {
      c: "#E05675",
      t: "Week 28 update",
      b: "Baby: broccoli (~37 cm) · Eyes open/close.",
      tm: "Yesterday, 12:00 PM",
    },
  ];

  const appointments = [
    {
      d: "28",
      m: "APR",
      c: "rose",
      t: "Routine antenatal check",
      sub: "Dr. Meena Rajan · Apollo Clinic",
      time: "10:30 AM · 45 min",
    },
    {
      d: "15",
      m: "MAY",
      c: "plum",
      t: "Growth scan ultrasound",
      sub: "Dr. Meena Rajan · Apollo Clinic",
      time: "2:00 PM · 60 min",
    },
  ];

  const milestones = [
    {
      e: "🎉",
      t: "Entered 3rd trimester",
      s: "Week 28 · Major milestone",
      c: "sage",
    },
    {
      e: "👀",
      t: "Baby's eyes opened",
      s: "Week 28 · Baby can detect light",
      c: "rose",
    },
    {
      e: "🫁",
      t: "Lung surfactant forming",
      s: "Week 28 · Delivery readiness",
      c: "plum",
    },
  ];

  return (
    <div className="space-y-5 animate-fade-up">
      <div className="grid gap-5 lg:grid-cols-2">
        {/* Recent Activity Column */}
        <div>
          <SectionHeader title="Recent activity" />
          <Card>
            {activities.map((item, idx) => (
              <div
                key={idx}
                className="flex gap-3 py-3"
                style={{
                  borderBottom:
                    idx < activities.length - 1 ? "1px solid #EAE4F5" : undefined,
                }}
              >
                <div className="flex flex-col items-center">
                  <div
                    className="w-2.5 h-2.5 rounded-full flex-shrink-0 mt-0.5"
                    style={{ background: item.c }}
                  />
                  {idx < activities.length - 1 && (
                    <div className="w-px flex-1 bg-border mt-1" />
                  )}
                </div>
                <div className="flex-1 pb-1">
                  <div className="text-[13px] font-semibold text-ink">
                    {item.t}
                  </div>
                  <div className="text-[12px] text-mid leading-relaxed mt-0.5">
                    {item.b}
                  </div>
                  <div className="text-[11px] text-muted mt-1">{item.tm}</div>
                </div>
              </div>
            ))}
          </Card>
        </div>

        {/* Right Column: Appointments & Milestones */}
        <div>
          <SectionHeader title="Upcoming appointments" />
          {appointments.map((appt, idx) => (
            <Card key={idx} className="flex gap-3 mb-3 !p-4">
              <div
                className={`w-12 h-12 rounded-xl flex flex-col items-center justify-center flex-shrink-0 ${
                  appt.c === "rose" ? "bg-rose-light" : "bg-plum-light"
                }`}
              >
                <div
                  className={`font-display text-xl font-bold leading-none ${
                    appt.c === "rose" ? "text-rose" : "text-plum"
                  }`}
                >
                  {appt.d}
                </div>
                <div
                  className={`text-[9px] font-bold ${
                    appt.c === "rose" ? "text-rose" : "text-plum"
                  }`}
                >
                  {appt.m}
                </div>
              </div>
              <div>
                <div className="text-[13px] font-bold text-ink">{appt.t}</div>
                <div className="text-[12px] text-mid mt-0.5">{appt.sub}</div>
                <div className="text-[11px] text-muted mt-0.5">{appt.time}</div>
              </div>
            </Card>
          ))}

          <SectionHeader title="Milestones" className="mt-2" />
          {milestones.map((ms, idx) => (
            <div
              key={idx}
              className={`flex items-center gap-3 p-3 rounded-xl border mb-2
                ${
                  ms.c === "sage"
                    ? "bg-sage-light border-sage/25"
                    : ms.c === "rose"
                    ? "bg-rose-light border-rose/25"
                    : "bg-plum-light border-plum/25"
                }`}
            >
              <span className="text-xl">{ms.e}</span>
              <div>
                <div
                  className={`text-[13px] font-semibold ${
                    ms.c === "sage"
                      ? "text-sage-dark"
                      : ms.c === "rose"
                      ? "text-rose-dark"
                      : "text-plum-dark"
                  }`}
                >
                  {ms.t}
                </div>
                <div className="text-[11px] text-muted">{ms.s}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
