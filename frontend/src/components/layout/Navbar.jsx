import React from "react";

export default function Navbar({ onOpenLogin }) {
  return (
    <nav
      className="fixed top-0 inset-x-0 z-50 border-b border-border"
      style={{
        background: "rgba(253,251,255,.9)",
        backdropFilter: "blur(16px)",
      }}
    >
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center gap-8">
        <div className="flex items-center gap-2.5 font-display text-xl font-bold text-ink">
          <div className="w-7 h-7 rounded-lg bg-rose flex items-center justify-center">
            <svg viewBox="0 0 20 20" width="14" fill="none">
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

        <div className="hidden md:flex gap-7 text-[14px] text-mid ml-4">
          <a href="#products" className="hover:text-ink transition-colors">
            Products
          </a>
          <a href="#how" className="hover:text-ink transition-colors">
            How it works
          </a>
          <a href="#mission" className="hover:text-ink transition-colors">
            Our mission
          </a>
        </div>

        <div className="ml-auto flex gap-3">
          <button
            onClick={onOpenLogin}
            className="px-4 py-2 rounded-full border border-border text-[13.5px] font-semibold text-mid hover:border-rose hover:text-rose transition-colors"
          >
            Log in
          </button>
          <button
            onClick={onOpenLogin}
            className="px-5 py-2 rounded-full bg-rose text-white text-[13.5px] font-semibold hover:bg-rose-dark transition-colors"
          >
            Request Demo
          </button>
        </div>
      </div>
    </nav>
  );
}
