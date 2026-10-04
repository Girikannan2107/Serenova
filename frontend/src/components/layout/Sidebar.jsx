import React from "react";

export default function Sidebar({
  isOpen,
  onClose,
  page,
  setPage,
  mode,
  setMode,
  week,
  setWeek,
  navItems,
  onExit,
}) {
  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 bg-ink/40 z-10 md:hidden"
          onClick={onClose}
        />
      )}
      <aside
        className={`
          ${isOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"}
          fixed md:relative top-0 left-0 h-screen w-60 bg-white border-r border-border
          flex flex-col z-20 transition-transform duration-300 flex-shrink-0
        `}
      >
        {/* Logo Section */}
        <div className="px-5 py-4 border-b border-border flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-rose flex items-center justify-center flex-shrink-0">
            <svg viewBox="0 0 20 20" width="16" fill="none">
              <circle cx="10" cy="10" r="7" stroke="white" strokeWidth="1.5" />
              <path
                d="M10 5.5C10 5.5 7 8 7 10.5C7 12 8.3 13 10 13S13 12 13 10.5C13 8 10 5.5 10 5.5Z"
                fill="white"
                opacity=".9"
              />
            </svg>
          </div>
          <div>
            <div className="font-display text-[16px] font-bold text-ink leading-none">
              Serenova
            </div>
            <div className="text-[10px] text-muted">Intelligent monitoring</div>
          </div>
        </div>

        {/* Mode Switcher */}
        <div className="px-4 py-3 border-b border-border">
          <div className="flex bg-[#F5F0FF] rounded-full p-0.5">
            <button
              onClick={() => {
                setMode("pregnancy");
                setPage("home");
              }}
              className={`flex-1 py-1.5 rounded-full text-[11px] font-semibold transition-all
                ${
                  mode === "pregnancy"
                    ? "bg-rose text-white shadow-sm"
                    : "text-muted hover:text-mid"
                }`}
            >
              🤰 Pregnancy
            </button>
            <button
              onClick={() => {
                setMode("postpartum");
                setPage("postpartum");
              }}
              className={`flex-1 py-1.5 rounded-full text-[11px] font-semibold transition-all
                ${
                  mode === "postpartum"
                    ? "text-white shadow-sm"
                    : "text-muted hover:text-mid"
                }`}
              style={mode === "postpartum" ? { background: "#E8916A" } : {}}
            >
              🌸 Postpartum
            </button>
          </div>
        </div>

        {/* Week Slider (Pregnancy Mode only) */}
        {mode === "pregnancy" && (
          <div className="px-4 py-3 border-b border-border">
            <div className="text-[10px] font-semibold text-muted mb-1.5 uppercase tracking-wider">
              Gestational week
            </div>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min="4"
                max="42"
                value={week}
                onChange={(e) => setWeek(+e.target.value)}
                className="flex-1 h-1.5 rounded-full appearance-none cursor-pointer accent-rose"
              />
              <span className="text-[12px] font-bold text-rose w-8 text-right">
                W{week}
              </span>
            </div>
          </div>
        )}

        {/* Nav Items */}
        <div className="flex-1 overflow-y-auto py-3 px-3">
          <div className="text-[10px] font-bold text-muted uppercase tracking-wider px-2 mb-2">
            Navigation
          </div>
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => {
                setPage(item.id);
                onClose();
              }}
              className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl cursor-pointer mb-0.5 text-left transition-all
                ${
                  page === item.id
                    ? "bg-rose-light text-rose font-semibold"
                    : "text-mid hover:bg-plum-light hover:text-plum"
                }`}
            >
              <span className="text-base w-5 text-center">{item.emoji}</span>
              <span className="text-[13.5px] font-medium flex-1">
                {item.label}
              </span>
              {item.badge && (
                <span className="text-[9px] font-bold bg-rose text-white px-1.5 py-0.5 rounded-full">
                  {item.badge}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* User Footer */}
        <div className="px-4 py-4 border-t border-border flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-full bg-rose-light flex items-center justify-center font-display text-[14px] font-bold text-rose flex-shrink-0">
            P
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-[13px] font-semibold text-ink truncate">
              Priya Sharma
            </div>
            <div className="text-[11px] text-muted">Due Jul 12, 2025</div>
          </div>
          <button
            onClick={onExit}
            className="text-[11px] text-muted hover:text-rose transition-colors flex-shrink-0"
          >
            Exit
          </button>
        </div>
      </aside>
    </>
  );
}
