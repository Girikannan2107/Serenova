import React, { useState } from "react";
import LandingPage from "./pages/LandingPage";
import DashboardPage from "./pages/DashboardPage";
import FetalMonitorPage from "./pages/FetalMonitorPage";
import VitalsTrendsPage from "./pages/VitalsTrendsPage";
import NutritionPage from "./pages/NutritionPage";
import EnvironmentPage from "./pages/EnvironmentPage";
import TimelinePage from "./pages/TimelinePage";
import DoctorPortalPage from "./pages/DoctorPortalPage";
import PostpartumPage from "./pages/PostpartumPage";
import AiAssistantPage from "./pages/AiAssistantPage";
import Sidebar from "./components/layout/Sidebar";
import TopHeader from "./components/layout/TopHeader";
import { initialAppData } from "./data/mockData";
import { useSensorData } from "./hooks/useSensorData";

const defaultNavItems = [
  { id: "home", emoji: "🏠", label: "Dashboard" },
  { id: "fetal", emoji: "💗", label: "Fetal Monitor", badge: "Live" },
  { id: "vitals", emoji: "📈", label: "Vitals Trends" },
  { id: "nutrition", emoji: "🥗", label: "Nutrition" },
  { id: "environment", emoji: "🌿", label: "Environment" },
  { id: "timeline", emoji: "📋", label: "Timeline" },
  { id: "doctor", emoji: "👩‍⚕️", label: "Doctor Portal" },
  { id: "ai", emoji: "✨", label: "AI Assistant" },
];

const pageTitleMap = {
  home: "Dashboard",
  fetal: "Fetal Monitor",
  vitals: "Vitals Trends",
  nutrition: "Nutrition",
  environment: "Environment",
  timeline: "Timeline",
  doctor: "Doctor Portal",
  postpartum: "Postpartum Mode",
  ai: "AI Assistant",
};

export default function App() {
  const [view, setView] = useState("landing"); // 'landing' | 'app'
  const [page, setPage] = useState("home");
  const [mode, setMode] = useState("pregnancy"); // 'pregnancy' | 'postpartum'
  const [week, setWeek] = useState(28);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [appData] = useState(initialAppData);

  // Live real sensor data stream hook
  const sensorState = useSensorData();

  const navItems =
    mode === "postpartum"
      ? [
          ...defaultNavItems.filter((m) => !["fetal", "vitals"].includes(m.id)),
          { id: "postpartum", emoji: "🌸", label: "Postpartum Mode" },
        ]
      : defaultNavItems;

  const renderActivePage = () => {
    if (mode === "postpartum" && page === "home") {
      return (
        <DashboardPage
          data={appData}
          week={week}
          setPage={setPage}
          sensorState={sensorState}
        />
      );
    }

    switch (page) {
      case "home":
        return (
          <DashboardPage
            data={appData}
            week={week}
            setPage={setPage}
            sensorState={sensorState}
          />
        );
      case "fetal":
        return (
          <FetalMonitorPage
            data={appData}
            sensorState={sensorState}
          />
        );
      case "vitals":
        return (
          <VitalsTrendsPage
            data={appData}
            sensorState={sensorState}
          />
        );
      case "nutrition":
        return <NutritionPage data={appData} />;
      case "environment":
        return <EnvironmentPage />;
      case "timeline":
        return <TimelinePage />;
      case "doctor":
        return (
          <DoctorPortalPage
            data={appData}
            sensorState={sensorState}
          />
        );
      case "postpartum":
        return <PostpartumPage data={appData} />;
      case "ai":
        return <AiAssistantPage data={appData} mode={mode} />;
      default:
        return (
          <DashboardPage
            data={appData}
            week={week}
            setPage={setPage}
            sensorState={sensorState}
          />
        );
    }
  };

  if (view === "landing") {
    return <LandingPage onLogin={() => setView("app")} />;
  }

  return (
    <div className="flex h-screen overflow-hidden bg-[#F5F0FF]">
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        page={page}
        setPage={setPage}
        mode={mode}
        setMode={setMode}
        week={week}
        setWeek={setWeek}
        navItems={navItems}
        onExit={() => setView("landing")}
      />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <TopHeader
          title={pageTitleMap[page] || "Dashboard"}
          week={week}
          mode={mode}
          riskScore={appData.risk.risk_score}
          riskLevel={appData.risk.risk_level}
          sensorState={sensorState}
          onOpenMobileMenu={() => setSidebarOpen((prev) => !prev)}
          onOpenAi={() => setPage("ai")}
        />

        <main className="flex-1 overflow-y-auto p-5">
          <div className="max-w-4xl mx-auto" key={page + mode}>
            {renderActivePage()}
          </div>
        </main>
      </div>
    </div>
  );
}
