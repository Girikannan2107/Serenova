/**
 * Serenova Clinical AI Companion Service (Demo / Mock Clinical Guidance)
 *
 * NOTE: Raw hardware sensor telemetry is not directly fed into automated diagnosis.
 * All clinical recommendations and chatbot responses use simulated educational context.
 */

export async function askSerenovaAi({ message, data, mode }) {
  // Simulate natural assistant latency without exposing external API credentials
  await new Promise((resolve) => setTimeout(resolve, 600));

  const msg = (message || "").toLowerCase();

  if (msg.includes("hrv") || msg.includes("declin")) {
    return "Your HRV has drifted downward from 78 ms to 66 ms over the last 5 days. In gestational tracking, this often reflects maternal fatigue or mild autonomic stress. Ensure adequate hydration and mention this trend to your care team.";
  }
  if (msg.includes("risk score") || msg.includes("32")) {
    return "A risk score of 32/100 sits in the mild/monitor range. Your fetal heart rate (137 bpm) and blood pressure (120/80) are reassuring, but routine glucose variation at 100 mg/dL should continue to be tracked.";
  }
  if (msg.includes("folate") || msg.includes("eat") || msg.includes("diet") || msg.includes("nutrition")) {
    return "Great sources of folate include dark leafy greens (spinach, methi), lentils, chickpeas, fortified cereals, and citrus fruits. Continue your prenatal vitamins as prescribed.";
  }
  if (msg.includes("heat") || msg.includes("37") || msg.includes("weather")) {
    return "A high heat index can increase maternal cardiovascular workload. Aim for 3 liters of water, stay in well-ventilated spaces, and avoid midday sun exposure.";
  }
  if (msg.includes("ppd") || msg.includes("edinburgh") || msg.includes("mood")) {
    return "Your Edinburgh score of 6/30 indicates low risk. Hormonal fluctuations are normal, but if feelings of overwhelm or sadness persist, please reach out to your healthcare provider.";
  }
  if (msg.includes("feed") || msg.includes("milk") || msg.includes("breast")) {
    return "Frequent on-demand feeding (8–12 times per 24 hours), adequate hydration, and skin-to-skin contact are proven ways to maintain a healthy milk supply.";
  }
  if (msg.includes("sensor") || msg.includes("pico") || msg.includes("hardware") || msg.includes("ecg") || msg.includes("ppg")) {
    return "The live hardware telemetry stream (ECG, PPG, IMU, Temperature) is capturing raw physical signals. Clinical diagnostic interpretations are separate and evaluated by licensed clinicians.";
  }

  return `Your ${mode === "postpartum" ? "postpartum recovery" : "pregnancy"} is progressing steadily. Primary indicators and vital trends remain within typical parameters. Always consult your obstetrician for medical decisions.`;
}
