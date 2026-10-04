import React, { useState } from "react";
import {
  StatusBadge,
  ProgressBar,
  MetricCard,
  SectionHeader,
  Card,
  MoodSelector,
  ApiResponseViewer,
} from "../components/common/CommonComponents";

export default function PostpartumPage({ data }) {
  const postpartum = data.postpartum;
  const [mood, setMood] = useState(postpartum.mood_today);
  const [showHelpline, setShowHelpline] = useState(false);
  const [feedLoggedToast, setFeedLoggedToast] = useState(false);
  const [selectedPain, setSelectedPain] = useState("");
  const [quickNote, setQuickNote] = useState("");

  const handleLogFeed = () => {
    setFeedLoggedToast(true);
    setTimeout(() => setFeedLoggedToast(false), 3500);
  };

  return (
    <div className="space-y-5 animate-fade-up">
      {/* Top Banner */}
      <div
        className="rounded-2xl p-4 flex items-center gap-4"
        style={{
          background: "linear-gradient(135deg,#FFF3ED,#FFF0F4)",
          border: "1px solid #F5D5C0",
        }}
      >
        <span className="text-3xl">🌸</span>
        <div className="flex-1">
          <div
            className="font-display text-[16px] font-semibold"
            style={{ color: "#993C1D" }}
          >
            Postpartum Mode · Day {postpartum.baby.age_days}
          </div>
          <div
            className="text-[12px] mt-0.5"
            style={{ color: "#D85A30" }}
          >
            You&apos;re doing amazingly. Your body is healing — be gentle with yourself.
          </div>
        </div>
        <div className="text-right">
          <div className="text-[11px] text-muted">Baby weight</div>
          <div className="font-display text-[18px] font-bold text-ink">
            {postpartum.baby.current_weight_kg} kg
          </div>
          <div className="text-[10px] text-sage">
            +
            {(
              (postpartum.baby.current_weight_kg -
                postpartum.baby.birth_weight_kg) *
              1000
            ).toFixed(0)}
            g since birth
          </div>
        </div>
      </div>

      {/* Maternal Vitals Postpartum */}
      <div>
        <SectionHeader title="Maternal vitals — postpartum" />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MetricCard
            icon="🩸"
            value={`${postpartum.bp_postpartum.systolic}/${postpartum.bp_postpartum.diastolic}`}
            unit="mmHg"
            label="Blood pressure"
            status="normal"
          />
          <MetricCard
            icon="⚖️"
            value={postpartum.baby.birth_weight_kg}
            unit="kg"
            label="Birth weight"
            status="normal"
            sub="Baby at birth"
          />
          <MetricCard
            icon="📉"
            value={`${Math.abs(postpartum.weight_change)} kg`}
            unit=""
            label="Weight lost"
            status="normal"
            sub="Since delivery"
          />
          <MetricCard
            icon="😴"
            value={postpartum.sleep_hours}
            unit="hrs"
            label="Last night sleep"
            status={postpartum.sleep_hours < 5 ? "watch" : "normal"}
          />
        </div>
      </div>

      {/* Mood and PPD Screening */}
      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <SectionHeader title="How are you feeling today?" />
          <MoodSelector value={mood} onChange={setMood} />
        </Card>

        <Card>
          <SectionHeader
            title="Edinburgh PPD screening"
            right={<StatusBadge status="normal" label="Low risk" />}
          />
          <div className="flex items-center gap-4 mb-3">
            <div className="text-center flex-shrink-0">
              <div className="font-display text-4xl font-bold text-sage">
                {postpartum.ppd_score}
              </div>
              <div className="text-[11px] text-muted">
                out of {postpartum.ppd_max}
              </div>
            </div>
            <div className="flex-1">
              <ProgressBar
                value={postpartum.ppd_score}
                max={postpartum.ppd_max}
                color="#2A9968"
                className="mb-1.5"
              />
              <div className="text-[11px] text-sage font-semibold">
                Score 0–9: low concern range
              </div>
            </div>
          </div>
          <p className="text-[12px] text-mid leading-relaxed mb-3">
            Mood trending positive. Next Edinburgh screening in 7 days. Track
            changes weekly.
          </p>
          <button
            onClick={() => setShowHelpline((prev) => !prev)}
            className="text-[12px] font-semibold text-rose bg-rose-light px-4 py-2 rounded-full hover:bg-rose-medium transition-colors"
          >
            {showHelpline ? "Hide resources" : "Talk to someone 💬"}
          </button>
          {showHelpline && (
            <div
              className="mt-3 p-3 rounded-xl text-[12px] leading-relaxed animate-fade-up"
              style={{
                background: "#fff0f0",
                border: "1px solid #f09595",
                color: "#993535",
              }}
            >
              <strong>iCall India:</strong> 9152987821 · Free counselling
              <br />
              <strong>Vandrevala Foundation:</strong> 1860-2662-345 · 24/7
              <br />
              <strong>NIMHANS:</strong> 080-46110007
              <br />
              <em className="block mt-2 opacity-80">
                PPD affects 1 in 7 mothers. You are not alone.
              </em>
            </div>
          )}
        </Card>
      </div>

      {/* Sleep and Lactation */}
      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <SectionHeader title="Sleep fragmentation" />
          <div className="text-[12px] text-muted mb-3">
            Last night ·{" "}
            {postpartum.sleep_blocks.filter((b) => b.type === "sleep").length}{" "}
            blocks · {postpartum.sleep_hours} hrs total
          </div>
          <div className="flex gap-1.5 mb-3">
            {postpartum.sleep_blocks.map((b, idx) => (
              <div
                key={idx}
                className="h-10 rounded-lg flex items-center justify-center text-[9px] font-bold text-white"
                style={{
                  flex: b.flex,
                  background: b.type === "sleep" ? "#7052B8" : "#EAE4F5",
                  color: b.type === "sleep" ? "white" : "#8E87A6",
                  minWidth: "28px",
                }}
              >
                {b.label}
              </div>
            ))}
          </div>
          <p className="text-[12px] text-mid leading-relaxed">
            Fragmented but within expected range for day {postpartum.baby.age_days}.
            Deep sleep ~15% — low but improving. Try sleeping when baby sleeps this
            afternoon.
          </p>
          <div className="mt-3 grid grid-cols-3 gap-2">
            {[
              { label: "Total sleep", val: `${postpartum.sleep_hours}h` },
              { label: "REM est.", val: "22%" },
              { label: "Deep sleep", val: "15%" },
            ].map((st) => (
              <div
                key={st.label}
                className="text-center p-2 bg-surface rounded-xl border border-border"
              >
                <div className="font-semibold text-ink text-[14px]">
                  {st.val}
                </div>
                <div className="text-[10px] text-muted">{st.label}</div>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <SectionHeader
            title="Lactation log"
            right={<StatusBadge status="normal" label="On track" />}
          />
          <div className="text-[12px] text-muted mb-3">
            Today: {postpartum.feeds_today} feeds · Last:{" "}
            {postpartum.last_feed_mins_ago} min ago
          </div>
          <div
            className="grid gap-1.5 mb-4"
            style={{ gridTemplateColumns: "repeat(6,1fr)" }}
          >
            {["6am", "9am", "12pm", "3pm", "6pm", "9pm"].map((t, idx) => (
              <div
                key={idx}
                className="aspect-square bg-rose rounded-xl flex items-center justify-center text-[9px] font-bold text-white"
              >
                {t}
              </div>
            ))}
          </div>
          <button
            onClick={handleLogFeed}
            className="w-full py-2.5 rounded-full bg-rose text-white text-[13px] font-semibold hover:bg-rose-dark transition-colors"
          >
            Log feed now
          </button>
          {feedLoggedToast && (
            <div className="mt-2 text-[12px] text-sage font-semibold text-center animate-fade-up">
              ✓ Feed logged at{" "}
              {new Date().toLocaleTimeString("en-GB", {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </div>
          )}
          <div className="mt-3 p-2.5 bg-rose-light rounded-xl text-[11px] text-mid">
            💡 Tip: Aim for 8–12 feeds per 24 hours. Baby&apos;s wet nappies (6+/day)
            confirm adequate intake.
          </div>
        </Card>
      </div>

      {/* Recovery Checklist */}
      <Card>
        <SectionHeader
          title={`Recovery checklist — Day ${postpartum.baby.age_days}`}
        />
        <div className="grid gap-2 sm:grid-cols-2">
          {[
            {
              label: "Lochia status",
              val: postpartum.lochia_status,
              icon: "🩸",
              note: "Normal colour & volume",
            },
            {
              label: "Incision / perineum",
              val: postpartum.incision_status,
              icon: "🩹",
              note: "Monitor for redness/swelling",
            },
            {
              label: "BP monitoring",
              val: "Daily",
              icon: "📊",
              note: `Last: ${postpartum.bp_postpartum.systolic}/${postpartum.bp_postpartum.diastolic} mmHg`,
            },
            {
              label: "Emotional check-in",
              val: mood || "Not logged",
              icon: "💭",
              note: "Edinburgh score: " + postpartum.ppd_score,
            },
          ].map((item) => (
            <div
              key={item.label}
              className="flex items-center gap-3 p-3 rounded-xl border border-border"
            >
              <span className="text-xl">{item.icon}</span>
              <div className="flex-1">
                <div className="text-[12px] font-semibold text-ink">
                  {item.label}
                </div>
                <div className="text-[11px] text-muted">{item.note}</div>
              </div>
              <span className="text-[12px] font-semibold text-sage capitalize">
                {item.val}
              </span>
            </div>
          ))}
        </div>
      </Card>

      {/* Baby Details Card */}
      <Card>
        <SectionHeader
          title={`Baby ${postpartum.baby.name} — Day ${postpartum.baby.age_days}`}
        />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            { label: "Age", val: `${postpartum.baby.age_days} days`, icon: "👶" },
            {
              label: "Birth weight",
              val: `${postpartum.baby.birth_weight_kg} kg`,
              icon: "⚖️",
            },
            {
              label: "Now",
              val: `${postpartum.baby.current_weight_kg} kg`,
              icon: "📈",
            },
            {
              label: "Weight gain",
              val: `+${(
                (postpartum.baby.current_weight_kg -
                  postpartum.baby.birth_weight_kg) *
                1000
              ).toFixed(0)}g`,
              icon: "✓",
            },
          ].map((stat) => (
            <div
              key={stat.label}
              className="text-center p-3 bg-surface rounded-xl border border-border"
            >
              <div className="text-xl mb-1">{stat.icon}</div>
              <div className="font-semibold text-ink text-[15px]">
                {stat.val}
              </div>
              <div className="text-[10px] text-muted mt-0.5">{stat.label}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Quick Notes & Symptoms */}
      <Card>
        <SectionHeader title="Quick notes & symptom log" />
        <div className="flex gap-2 mb-3">
          {["No pain", "Mild discomfort", "Moderate pain", "Severe"].map(
            (level) => (
              <button
                key={level}
                onClick={() => setSelectedPain(level)}
                className={`flex-1 py-2 rounded-xl text-[11px] font-semibold border transition-all
                  ${
                    selectedPain === level
                      ? "bg-rose text-white border-rose"
                      : "bg-surface text-muted border-border hover:border-rose"
                  }`}
              >
                {level}
              </button>
            )
          )}
        </div>
        <textarea
          value={quickNote}
          onChange={(e) => setQuickNote(e.target.value)}
          rows={3}
          placeholder="Note symptoms, feelings, questions for your doctor…"
          className="w-full px-3.5 py-2.5 border border-border rounded-xl text-[13px] font-body outline-none resize-none text-ink focus:border-rose transition-colors"
        />
        <button className="mt-2 px-5 py-2 rounded-full bg-ink text-white text-[12px] font-semibold hover:opacity-90 transition-opacity">
          Save note
        </button>
      </Card>

      <ApiResponseViewer data={{ postpartum: data.postpartum }} />
    </div>
  );
}
