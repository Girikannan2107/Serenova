import React, { useState } from "react";
import Navbar from "../components/layout/Navbar";
import LoginModal from "../components/modals/LoginModal";
import FetusModel3D from "../components/3d/FetusModel3D";
import { productsData, howItWorksData } from "../data/mockData";

export default function LandingPage({ onLogin }) {
  const [showLoginModal, setShowLoginModal] = useState(false);

  return (
    <div className="min-h-screen font-body bg-surface">
      <Navbar onOpenLogin={() => setShowLoginModal(true)} />

      {/* Hero Section */}
      <div
        className="pt-16 min-h-screen flex items-center"
        style={{
          background:
            "linear-gradient(135deg,#FFF5F7 0%,#F5F0FF 55%,#FFF8F0 100%)",
        }}
      >
        <div className="max-w-6xl mx-auto px-6 py-20 grid md:grid-cols-2 gap-12 items-center">
          <div>
            <div className="inline-flex items-center gap-2 bg-rose-light border border-rose-medium rounded-full px-4 py-1.5 text-[12px] font-semibold text-rose mb-6">
              <span className="w-1.5 h-1.5 rounded-full bg-rose animate-blink" />
              AI-powered fetal & maternal monitoring
            </div>
            <h1 className="font-display text-5xl md:text-6xl font-bold text-ink leading-tight mb-5">
              Pregnancy monitoring,{" "}
              <em className="text-rose">redefined</em>
            </h1>
            <p className="text-[16px] text-mid leading-relaxed mb-8">
              Serenova combines medical-grade wearables, CNN-powered fetal
              analysis and real-time doctor connectivity into one intelligent
              platform — built to reach mothers in primary health centres and
              rural communities.
            </p>
            <div className="flex gap-3 flex-wrap mb-8">
              <button
                onClick={() => setShowLoginModal(true)}
                className="px-7 py-3.5 rounded-full bg-rose text-white font-semibold hover:bg-rose-dark transition-all hover:-translate-y-0.5"
              >
                Start monitoring →
              </button>
              <button
                onClick={() =>
                  document
                    .getElementById("how")
                    ?.scrollIntoView({ behavior: "smooth" })
                }
                className="px-7 py-3.5 rounded-full border border-border text-mid font-semibold hover:border-plum hover:text-plum transition-colors"
              >
                How it works
              </button>
            </div>
            <div className="flex flex-wrap gap-2.5">
              {[
                "Early-stage · research-backed",
                "Built for rural deployment",
                "AI risk scoring with CNN",
                "No inflated metrics",
              ].map((badge, i) => (
                <span
                  key={i}
                  className="text-[11px] font-semibold text-muted bg-white border border-border px-3 py-1 rounded-full"
                >
                  {badge}
                </span>
              ))}
            </div>
          </div>

          <div className="relative h-80 md:h-[480px]">
            <div
              className="absolute inset-0 rounded-3xl overflow-hidden"
              style={{
                background: "linear-gradient(135deg,#FFE8ED,#F0E8FF)",
              }}
            >
              <FetusModel3D week={28} />
              <div className="animate-float absolute bottom-6 right-6 bg-white rounded-2xl p-3 shadow-float">
                <div className="text-[10px] font-semibold text-muted uppercase">
                  Fetal HR · Live
                </div>
                <div className="font-display text-2xl font-bold text-rose">
                  137{" "}
                  <span className="text-[13px] text-muted font-body font-normal">
                    bpm
                  </span>
                </div>
                <div className="text-[11px] text-sage font-semibold">
                  ✓ Healthy
                </div>
              </div>

              <div
                className="animate-float absolute top-6 left-6 bg-white rounded-2xl p-3 shadow-float"
                style={{ animationDelay: "1.5s" }}
              >
                <div className="text-[10px] font-semibold text-muted uppercase">
                  Risk score
                </div>
                <div className="font-display text-2xl font-bold text-plum">
                  32{" "}
                  <span className="text-[13px] text-muted font-body font-normal">
                    /100
                  </span>
                </div>
                <div className="text-[11px] text-sage font-semibold">
                  Low risk
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Who We Serve Section */}
      <div className="max-w-6xl mx-auto px-6 py-20">
        <div className="text-[11px] font-semibold text-rose uppercase tracking-widest mb-2">
          Who we serve
        </div>
        <h2 className="font-display text-4xl font-bold text-ink mb-3">
          A New Age Fetal-Maternal Monitoring System
        </h2>
        <p className="text-mid max-w-xl mb-12 leading-relaxed">
          Designed to bring clinical-grade monitoring to every mother — from
          urban hospitals to rural primary health centres — where access to
          obstetric care is limited and every alert counts.
        </p>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Card for Mothers */}
          <div
            className="rounded-3xl overflow-hidden border border-rose-medium"
            style={{
              background: "linear-gradient(135deg,#FFF0F4,#FFE8ED)",
            }}
          >
            <div className="h-56 relative flex items-center justify-center overflow-hidden">
              <FetusModel3D week={28} />
              <div className="absolute bottom-4 right-4 bg-white rounded-xl p-2.5 shadow-card text-[12px]">
                <div className="font-semibold text-rose">
                  Week 28 · Broccoli 🥦
                </div>
                <div className="text-muted">~1.1 kg · Eyes opening</div>
              </div>
            </div>
            <div className="p-6">
              <div className="text-[11px] font-bold text-rose uppercase tracking-widest mb-2">
                FOR MOTHERS
              </div>
              <h3 className="font-display text-2xl font-bold text-ink mb-3">
                Serenova for Mothers
              </h3>
              <p className="text-[14px] text-mid leading-relaxed mb-4">
                Peace of mind through continuous monitoring. Know your baby is
                healthy. Catch anomalies before they become emergencies — even
                at home, in a village, or far from the nearest hospital.
              </p>
              <button
                onClick={onLogin}
                className="px-5 py-2.5 rounded-full bg-rose text-white text-[13.5px] font-semibold hover:bg-rose-dark transition-colors"
              >
                Open Mother App →
              </button>
            </div>
          </div>

          {/* Card for Hospitals & Clinics */}
          <div
            className="rounded-3xl overflow-hidden border border-plum-medium"
            style={{
              background: "linear-gradient(135deg,#F0EBFF,#EDE8FF)",
            }}
          >
            <div className="h-56 bg-plum-light relative overflow-hidden flex items-center justify-center p-6">
              <div className="bg-white rounded-2xl w-full p-4 shadow-card">
                <div className="text-[11px] font-bold text-ink mb-2">
                  Priya S. · W28 · Low risk
                </div>
                <div className="h-1.5 bg-plum-medium rounded-full mb-1.5 w-full" />
                <div className="h-1.5 bg-border rounded-full mb-3 w-2/3" />
                <div className="bg-plum-light rounded-xl p-3 text-[13px] font-semibold text-plum text-center mb-2">
                  BP 119/74
                </div>
                <div className="flex gap-2">
                  <div className="flex-1 py-1.5 bg-plum rounded-lg text-[11px] font-semibold text-white text-center">
                    Export PDF
                  </div>
                  <div className="flex-1 py-1.5 bg-plum-light border border-plum/30 rounded-lg text-[11px] font-semibold text-plum text-center">
                    Add Note
                  </div>
                </div>
                <div className="absolute top-6 right-6 w-12 h-12 rounded-full bg-plum flex flex-col items-center justify-center shadow-float">
                  <div className="text-white text-[11px] font-bold leading-none">
                    82%
                  </div>
                  <div className="text-white text-[8px] opacity-75">risk</div>
                </div>
              </div>
            </div>
            <div className="p-6">
              <div className="text-[11px] font-bold text-plum-dark uppercase tracking-widest mb-2">
                FOR HOSPITALS & CLINICS
              </div>
              <h3 className="font-display text-2xl font-bold text-ink mb-3">
                Serenova for Hospitals & Clinics
              </h3>
              <p className="text-[14px] text-mid leading-relaxed mb-4">
                Live patient dashboards. CNN hypoxia scoring. Drift anomaly
                detection across full patient lists. Remote monitoring for
                high-risk pregnancies — including from primary health centres
                with limited on-site staff.
              </p>
              <button
                onClick={onLogin}
                className="px-5 py-2.5 rounded-full bg-plum text-white text-[13.5px] font-semibold hover:bg-plum-dark transition-colors"
              >
                See Hospital View →
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Product Suite Section */}
      <div id="products" className="py-20" style={{ background: "#F5F0FF" }}>
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-[11px] font-semibold text-rose uppercase tracking-widest mb-2">
            Our Devices & App
          </div>
          <h2 className="font-display text-4xl font-bold text-ink mb-3">
            The Serenova product suite
          </h2>
          <p className="text-mid max-w-xl mb-12 leading-relaxed">
            Two purpose-built wearables and one AI app — designed for real
            clinical environments and last-mile rural health deployment.
          </p>

          <div className="grid md:grid-cols-3 gap-6">
            {productsData.map((prod, i) => (
              <div
                key={i}
                className="bg-white rounded-3xl border border-border overflow-hidden hover:shadow-float hover:-translate-y-1 transition-all duration-200"
              >
                <div
                  className={`h-48 flex items-center justify-center text-6xl ${prod.bg}`}
                >
                  {prod.icon}
                </div>
                <div className="p-6">
                  <div className="text-[10px] font-bold text-muted uppercase tracking-wider mb-2 leading-tight">
                    {prod.tag}
                  </div>
                  <div className="font-display text-xl font-bold text-ink mb-2">
                    {prod.name}
                  </div>
                  <div className="text-[13px] text-mid leading-relaxed mb-4">
                    {prod.desc}
                  </div>
                  <ul className="space-y-1.5 mb-5">
                    {prod.features.map((feat, fi) => (
                      <li
                        key={fi}
                        className="flex items-start gap-2 text-[12px] text-mid"
                      >
                        <span className="text-rose mt-0.5 flex-shrink-0">
                          •
                        </span>
                        {feat}
                      </li>
                    ))}
                  </ul>
                  <button
                    onClick={prod.isApp ? onLogin : undefined}
                    className={`text-[13px] font-semibold flex items-center gap-1 transition-all
                      ${
                        prod.isApp
                          ? "text-rose hover:gap-2 cursor-pointer"
                          : "text-plum hover:gap-2"
                      }`}
                  >
                    {prod.isApp ? "Open App →" : "Know More →"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* How It Works Section */}
      <div
        id="how"
        className="py-20"
        style={{ background: "linear-gradient(135deg,#FFF0F4,#F0EBFF)" }}
      >
        <div className="max-w-6xl mx-auto px-6 text-center">
          <div className="text-[11px] font-semibold text-rose uppercase tracking-widest mb-2">
            Simple process
          </div>
          <h2 className="font-display text-4xl font-bold text-ink mb-3">
            How Serenova works
          </h2>
          <p className="text-mid mb-14">
            From setup to your first AI insight in under 5 minutes.
          </p>

          <div className="relative">
            <div
              className="hidden md:block absolute h-px bg-gradient-to-r from-rose to-plum z-0"
              style={{ top: "32px", left: "12.5%", right: "12.5%" }}
            />
            <div className="grid md:grid-cols-4 gap-8 relative z-10">
              {howItWorksData.map((step, i) => (
                <div key={i} className="text-center">
                  <div
                    className="w-16 h-16 rounded-full bg-white mx-auto mb-4 flex items-center justify-center font-display text-2xl font-bold"
                    style={{
                      border: `2.5px solid ${i < 2 ? "#E05675" : "#7052B8"}`,
                      color: i < 2 ? "#E05675" : "#7052B8",
                    }}
                  >
                    {step.n}
                  </div>
                  <div className="text-2xl mb-2">{step.e}</div>
                  <div className="font-display text-[15px] font-semibold text-ink mb-2">
                    {step.t}
                  </div>
                  <div className="text-[12px] text-mid leading-relaxed">
                    {step.d}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Mission Section */}
      <div id="mission" className="py-20 bg-ink text-white">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <div className="text-[11px] font-semibold text-rose uppercase tracking-widest mb-4">
            Our mission
          </div>
          <h2 className="font-display text-4xl md:text-5xl font-bold mb-6">
            Maternal mortality is <em className="text-rose">preventable</em>
          </h2>
          <p
            className="text-[16px] leading-relaxed mb-8 max-w-2xl mx-auto"
            style={{ color: "rgba(255,255,255,.7)" }}
          >
            Most high-risk pregnancy complications — pre-eclampsia, fetal
            hypoxia, gestational diabetes — are detectable early if you have the
            right tools. We are building Serenova to put those tools in the hands
            of every mother and every primary health worker, not just those in
            well-equipped urban hospitals.
          </p>
          <div className="flex flex-wrap gap-3 justify-center mb-10">
            {[
              "Early-stage startup",
              "PHC kit deployment planned",
              "CNN fetal AI model",
              "International competition build",
            ].map((tag, i) => (
              <span
                key={i}
                className="text-[11px] font-semibold px-4 py-2 rounded-full"
                style={{
                  color: "rgba(255,255,255,.6)",
                  background: "rgba(255,255,255,.08)",
                }}
              >
                {tag}
              </span>
            ))}
          </div>
          <button
            onClick={onLogin}
            className="px-8 py-4 rounded-full bg-rose text-white font-semibold text-[15px] hover:bg-rose-dark transition-all hover:-translate-y-0.5"
          >
            Try Serenova →
          </button>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-white border-t border-border py-10">
        <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 font-display text-lg font-bold text-ink">
            <div className="w-6 h-6 rounded-lg bg-rose flex items-center justify-center">
              <svg viewBox="0 0 20 20" width="12" fill="none">
                <circle cx="10" cy="10" r="7" stroke="white" strokeWidth="1.5" />
                <path
                  d="M10 5.5C10 5.5 7 8 7 10.5C7 12 8.3 13 10 13S13 12 13 10.5C13 8 10 5.5 10 5.5Z"
                  fill="white"
                  opacity=".9"
                />
              </svg>
            </div>
            Serenova
          </div>
          <div className="text-[12px] text-muted text-center">
            Intelligent maternal monitoring · Early-stage product · For
            demonstration & competition purposes
          </div>
          <div className="flex gap-4 text-[12px] text-muted">
            <a href="#" className="hover:text-ink transition-colors">
              Privacy
            </a>
            <a href="#" className="hover:text-ink transition-colors">
              Contact
            </a>
          </div>
        </div>
      </footer>

      <LoginModal
        isOpen={showLoginModal}
        onClose={() => setShowLoginModal(false)}
        onLogin={onLogin}
      />
    </div>
  );
}
