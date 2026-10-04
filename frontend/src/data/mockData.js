export const initialAppData = {
  fetal: {
    gestational_week: 28,
    cnn_raw: {
      hypoxia_probability: 0.18,
      hypoxia_flag: false,
    },
    fetal_status: "normal",
    fhr: 137,
    movement_count: 20,
    variability: 11,
  },
  maternal: {
    overall_status: "watch",
    metrics: {
      heart_rate: { value: 90, status: "normal" },
      spo2: { value: 98, status: "normal" },
      blood_pressure: { systolic: 120, diastolic: 80, status: "normal" },
      hrv: { value: 66, status: "watch" },
      glucose: { value: 100, status: "watch" },
      temperature: { value: 36.8, status: "normal" },
      weight: { value: 67.9, status: "normal" },
      respiratory_rate: { value: 18, status: "normal" },
    },
    drift_anomalies: ["HRV declining 5-day trend: 78→66 ms"],
  },
  risk: {
    risk_score: 32,
    risk_level: "medium",
    alert_tone: "watchful",
    positives: [
      "CNN hypoxia probability is 0.18 (low risk)",
      "FHR is 140 bpm within normal range",
      "Blood pressure stable – 21 consecutive days",
      "SpO₂ 98% – excellent oxygenation",
    ],
    watchful_items: [
      "Glucose level is 100 (borderline – monitor closely)",
      "HRV declining trend – possible fatigue or early anaemia",
    ],
    alert_items: [],
    conditions: [
      { name: "Gestational diabetes (GDM)", status: "watch" },
      { name: "Hypoglycemia", status: "low" },
      { name: "Preeclampsia", status: "low" },
      { name: "Anaemia", status: "low" },
      { name: "Fetal hypoxia", status: "low" },
    ],
    plain_language_summary:
      "Your baby is doing well overall. Some mild glucose variation needs monitoring.",
    doctor_summary:
      "28-week pregnancy. Stable vitals. Mild glucose elevation. CNN hypoxia 0.18 – low risk. HRV drift noted.",
    rag: {
      summary: "Risk score is 32. Conditions include GDM (watch).",
      clinical_findings: [
        "Low SpO₂ can indicate anaemia or hypoxia",
        "Elevated glucose suggests gestational diabetes risk",
      ],
      explanation:
        "WHO guidelines suggest monitoring glucose levels closely during second trimester.",
      decision: "Monitor glucose, no emergency",
    },
  },
  nutrition: {
    compliance_score: 75,
    deficiencies: ["vitamin D", "iron"],
    recommendations: [
      { nutrient: "iron", suggestion: "Increase leafy greens and supplements" },
      {
        nutrient: "vitamin D",
        suggestion: "15 min morning sun or supplement 600 IU",
      },
    ],
  },
  labor: {
    active: false,
  },
  postpartum: {
    overall_status: "normal",
    mood_today: "okay",
    ppd_score: 6,
    ppd_max: 30,
    sleep_status: "fragmented",
    sleep_hours: 5.5,
    sleep_blocks: [
      { label: "11 PM–1 AM", type: "sleep", flex: 2 },
      { label: "Feed", type: "feed", flex: 0.5 },
      { label: "2–3:30 AM", type: "sleep", flex: 1.5 },
      { label: "Feed", type: "feed", flex: 0.5 },
      { label: "4:30–6:30", type: "sleep", flex: 2 },
    ],
    feeds_today: 6,
    last_feed_mins_ago: 83,
    lochia_status: "normal",
    incision_status: "healing",
    bp_postpartum: { systolic: 118, diastolic: 76 },
    weight_change: -4.2,
    baby: {
      name: "Baby Sharma",
      birth_weight_kg: 3.1,
      current_weight_kg: 3.28,
      age_days: 14,
    },
  },
  conversation: {
    response:
      "Your pregnancy is progressing well. Keep monitoring glucose levels.",
    red_flag_detected: false,
    suggested_follow_ups: [
      "Check glucose weekly",
      "Maintain diet plan",
    ],
  },
};

export const productsData = [
  {
    tag: "SERENOVA BAND — MATERNAL WRISTBAND",
    name: "Serenova Band",
    icon: "⌚",
    bg: "bg-rose-light",
    desc: "The Serenova Band is a medical-grade maternal wristband worn 24/7 by the mother. It continuously captures heart rate, HRV, SpO₂, skin temperature, blood pressure trend, and movement patterns — all streamed to the app.",
    features: [
      "Continuous HR, HRV, SpO₂, temperature monitoring",
      "Detects drift anomalies across vital trends",
      "Bluetooth 5.0 · 7-day battery · IPX7 waterproof",
      "Worn on the wrist — comfortable for daily wear",
      "Syncs to Serenova app in real time",
    ],
  },
  {
    tag: "SERENOVA STRAP — FETAL MONITORING BELT",
    name: "Serenova Strap",
    icon: "🫀",
    bg: "bg-plum-light",
    desc: "The Serenova Strap is a soft abdominal sensor belt worn by the mother from week 28 onwards. It performs medical-grade Non-Stress Test (NST / CTG) monitoring, capturing fetal heart rate and uterine activity — streamed live to the doctor.",
    features: [
      "Medical-grade NST / CTG fetal monitoring at home",
      "Captures fetal HR, variability, and movement count",
      "CNN fetal hypoxia probability analysis in real time",
      "Streams live CTG to your doctor's portal",
      "Doctor-reviewed report generated within 2 hours",
      "Soft, adjustable belt — use from week 28 onwards",
    ],
  },
  {
    tag: "SERENOVA APP — AI PREGNANCY COMPANION",
    name: "Serenova App",
    icon: "📱",
    bg: "bg-gradient-to-br from-rose-light to-plum-light",
    desc: "The intelligent hub connecting both Serenova devices. AI-powered risk scoring, CNN fetal hypoxia analysis, nutrition tracking, drift anomaly detection, postpartum mode and a live doctor portal — all in one place.",
    features: [
      "CNN fetal hypoxia probability scoring",
      "Drift anomaly detection across all vitals",
      "Pregnancy & postpartum switching modes",
      "Real-time doctor portal with PDF export",
      "AI assistant powered by Claude",
    ],
    isApp: true,
  },
];

export const howItWorksData = [
  {
    n: "1",
    e: "📲",
    t: "Download & Sign Up",
    d: "Create your profile with due date, gestational week, and health conditions. No fake credentials needed — you're in control of your data.",
  },
  {
    n: "2",
    e: "⌚",
    t: "Wear the Band & Strap",
    d: "Put on the Serenova Band (wrist) and connect the Serenova Strap (abdomen from week 28). Both sync automatically via Bluetooth.",
  },
  {
    n: "3",
    e: "✨",
    t: "Get AI Insights",
    d: "Serenova AI analyses vital trends, flags drift anomalies, and explains risk scores in plain language — never just a number.",
  },
  {
    n: "4",
    e: "👩‍⚕️",
    t: "Doctor-Connected",
    d: "Your doctor receives a live clinical dashboard with full vital history, CNN fetal scores, and one-tap PDF reports.",
  },
];

export const environmentalData = {
  aqi: {
    value: 142,
    label: "AQI – Coimbatore",
    sub: "Unhealthy for sensitive groups",
    advice:
      "Limit outdoor activity to 15 min. PM₂.₅ at this level can cross the placenta. Use indoor air purifier.",
    color: "amber",
    icon: "💨",
  },
  uv: {
    value: 8,
    label: "UV Index",
    sub: "Very High",
    advice:
      "Apply SPF 50+ if going outside. UV worsens pregnancy melasma. Avoid 10 AM–3 PM.",
    color: "rose",
    icon: "☀️",
  },
  heat: {
    value: "31°C",
    label: "Heat Index",
    sub: "Feels like 36°C",
    advice:
      "Target 3 L water today. Avoid prolonged standing. Cool baths help lower core temp safely.",
    color: "sage",
    icon: "🌡️",
  },
};
