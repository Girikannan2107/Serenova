import React, { useState } from "react";
import {
  StatusBadge,
  ProgressBar,
  SectionHeader,
  Card,
  HighlightCard,
  ApiResponseViewer,
} from "../components/common/CommonComponents";

export default function NutritionPage({ data }) {
  const nutrition = data.nutrition;

  const initialMeals = [
    {
      e: "🥣",
      name: "Oatmeal with chia & berries",
      cal: "Breakfast · 380 kcal · High folate, iron",
      tag: "Breakfast",
      tc: "bg-sage-light text-sage-dark",
    },
    {
      e: "🥗",
      name: "Lentil & spinach salad with feta",
      cal: "Lunch · 520 kcal · Iron, calcium, protein",
      tag: "Lunch",
      tc: "bg-plum-light text-plum-dark",
    },
    {
      e: "🍊",
      name: "Orange & almonds",
      cal: "Snack · 180 kcal · Vitamin C boosts iron",
      tag: "Snack",
      tc: "bg-amber-light text-amber-dark",
    },
    {
      e: "🐟",
      name: "Baked salmon with quinoa & broccoli",
      cal: "Dinner · 600 kcal · DHA omega-3, protein",
      tag: "Dinner",
      tc: "bg-rose-light text-rose-dark",
    },
  ];

  const alternativeMeals = [
    {
      e: "🥣",
      name: "Greek yogurt parfait with mango",
      cal: "Breakfast · 360 kcal · Calcium, probiotics",
      tag: "Breakfast",
      tc: "bg-sage-light text-sage-dark",
    },
    {
      e: "🍱",
      name: "Chickpea & sweet potato curry",
      cal: "Lunch · 540 kcal · Iron, folate, fibre",
      tag: "Lunch",
      tc: "bg-plum-light text-plum-dark",
    },
    {
      e: "🥜",
      name: "Edamame & walnut mix",
      cal: "Snack · 200 kcal · Omega-3, protein",
      tag: "Snack",
      tc: "bg-amber-light text-amber-dark",
    },
    {
      e: "🥦",
      name: "Stir-fried tofu, broccoli noodles",
      cal: "Dinner · 580 kcal · Calcium, iron, folate",
      tag: "Dinner",
      tc: "bg-rose-light text-rose-dark",
    },
  ];

  const [meals, setMeals] = useState(initialMeals);
  const [isSpinning, setIsSpinning] = useState(false);

  const handleRegenerate = () => {
    setIsSpinning(true);
    setTimeout(() => {
      setMeals((current) =>
        current === initialMeals ? alternativeMeals : initialMeals
      );
      setIsSpinning(false);
    }, 700);
  };

  const macros = [
    { c: "#E05675", n: "Protein", v: "68g", pct: 75 },
    { c: "#7052B8", n: "Carbs", v: "195g", pct: 65 },
    { c: "#2A9968", n: "Fats", v: "52g", pct: 58 },
    { c: "#C48A08", n: "Fibre", v: "18g", pct: 45 },
  ];

  const nutrients = [
    { i: "🫀", name: "Iron", v: "27mg", g: "27mg", pct: 100, ok: true },
    { i: "🧠", name: "Folate", v: "520mcg", g: "600mcg", pct: 87, ok: false },
    { i: "🦴", name: "Calcium", v: "900mg", g: "1000mg", pct: 90, ok: false },
    { i: "🐟", name: "DHA Omega-3", v: "180mg", g: "200mg", pct: 90, ok: false },
    { i: "☀️", name: "Vitamin D", v: "280 IU", g: "600 IU", pct: 47, ok: false },
    { i: "💧", name: "Hydration", v: "1.9 L", g: "2.5 L", pct: 76, ok: false },
  ];

  return (
    <div className="space-y-5 animate-fade-up">
      <div className="grid gap-5 lg:grid-cols-2">
        {/* Today's Macros */}
        <Card>
          <SectionHeader
            title="Today's macros"
            right={
              <StatusBadge
                status="low"
                label={`${nutrition.compliance_score}/100 score`}
              />
            }
          />
          <div className="flex items-center gap-4 mb-4">
            <svg width="90" height="90" viewBox="0 0 90 90" className="flex-shrink-0">
              <circle
                cx="45"
                cy="45"
                r="34"
                fill="none"
                stroke="#EAE4F5"
                strokeWidth="7"
              />
              <circle
                cx="45"
                cy="45"
                r="34"
                fill="none"
                stroke="#E05675"
                strokeWidth="7"
                strokeDasharray="80 134"
                strokeDashoffset="25"
                strokeLinecap="round"
              />
              <circle
                cx="45"
                cy="45"
                r="34"
                fill="none"
                stroke="#7052B8"
                strokeWidth="7"
                strokeDasharray="50 134"
                strokeDashoffset="-55"
                strokeLinecap="round"
              />
              <circle
                cx="45"
                cy="45"
                r="34"
                fill="none"
                stroke="#2A9968"
                strokeWidth="7"
                strokeDasharray="38 134"
                strokeDashoffset="-105"
                strokeLinecap="round"
              />
              <text
                x="45"
                y="42"
                textAnchor="middle"
                fontSize="12"
                fontWeight="700"
                fill="#16121E"
              >
                1,680
              </text>
              <text
                x="45"
                y="54"
                textAnchor="middle"
                fontSize="8"
                fill="#8E87A6"
              >
                cal today
              </text>
            </svg>

            <div className="flex-1 space-y-2.5">
              {macros.map((m) => (
                <div key={m.n}>
                  <div className="flex justify-between text-[12px] mb-1">
                    <span className="text-mid flex items-center gap-1.5">
                      <span
                        className="w-2 h-2 rounded-full flex-shrink-0"
                        style={{ background: m.c }}
                      />
                      {m.n}
                    </span>
                    <span className="font-semibold text-ink">{m.v}</span>
                  </div>
                  <ProgressBar value={m.pct} color={m.c} />
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* Key Nutrients */}
        <Card>
          <SectionHeader title="Key nutrients — Week 28" />
          {nutrients.map((nItem, idx) => (
            <div
              key={nItem.name}
              className={`flex items-center gap-3 py-2.5 ${
                idx < nutrients.length - 1 ? "border-b border-border" : ""
              }`}
            >
              <span className="text-base">{nItem.i}</span>
              <div className="flex-1">
                <div className="flex justify-between text-[12px] mb-0.5">
                  <span className="font-semibold text-ink">{nItem.name}</span>
                  <span className="text-muted">
                    {nItem.v} / {nItem.g}
                  </span>
                </div>
                <ProgressBar
                  value={nItem.pct}
                  color={nItem.ok ? "#2A9968" : "#C48A08"}
                />
              </div>
              <span
                className={`text-[11px] font-semibold px-2 py-0.5 rounded-full ${
                  nItem.ok
                    ? "bg-sage-light text-sage-dark"
                    : "bg-amber-light text-amber-dark"
                }`}
              >
                {nItem.pct}%
              </span>
            </div>
          ))}

          {nutrition.deficiencies.length > 0 && (
            <div className="mt-3 p-2.5 bg-amber-light border border-amber/20 rounded-xl text-[11px] text-amber-dark">
              ⚠ Deficiencies detected: {nutrition.deficiencies.join(", ")}
            </div>
          )}
        </Card>
      </div>

      {/* Meal Suggestions */}
      <div>
        <SectionHeader
          title="AI meal suggestions"
          right={
            <button
              onClick={handleRegenerate}
              className="flex items-center gap-1.5 text-[12px] font-semibold text-rose bg-rose-light px-3 py-1.5 rounded-full hover:bg-rose-medium transition-colors"
            >
              <span className={isSpinning ? "animate-spin-slow inline-block" : ""}>
                ✨
              </span>{" "}
              Regenerate
            </button>
          }
        />
        <div className="space-y-2">
          {meals.map((meal, idx) => (
            <div
              key={idx}
              className="flex items-center gap-3 p-3.5 bg-white border border-border rounded-xl hover:border-rose-medium transition-colors"
            >
              <span className="text-2xl w-9 text-center">{meal.e}</span>
              <div className="flex-1">
                <div className="text-[13px] font-semibold text-ink">
                  {meal.name}
                </div>
                <div className="text-[11px] text-muted mt-0.5">{meal.cal}</div>
              </div>
              <span
                className={`text-[11px] font-semibold px-2.5 py-1 rounded-full ${meal.tc}`}
              >
                {meal.tag}
              </span>
            </div>
          ))}
        </div>

        {nutrition.recommendations.length > 0 && (
          <HighlightCard className="mt-4">
            <div className="font-display text-[13px] font-semibold text-plum-dark mb-2">
              AI recommendations
            </div>
            {nutrition.recommendations.map((rec, idx) => (
              <div key={idx} className="flex gap-2 text-[13px] text-mid">
                <span className="text-rose">→</span>
                <span>
                  <strong className="text-ink capitalize">{rec.nutrient}:</strong>{" "}
                  {rec.suggestion}
                </span>
              </div>
            ))}
          </HighlightCard>
        )}
      </div>

      <ApiResponseViewer data={{ nutrition: data.nutrition }} />
    </div>
  );
}
