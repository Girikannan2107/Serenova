import React from "react";

export default function LoginModal({ isOpen, onClose, onLogin }) {
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-5"
      style={{ background: "rgba(22,18,30,.6)" }}
    >
      <div className="bg-white rounded-3xl w-full max-w-sm overflow-hidden shadow-float animate-fade-up">
        <div
          className="p-8 text-center"
          style={{ background: "linear-gradient(135deg,#FFF0F4,#F0EBFF)" }}
        >
          <div className="w-14 h-14 rounded-2xl bg-rose mx-auto mb-4 flex items-center justify-center">
            <svg viewBox="0 0 28 28" width="28" fill="none">
              <circle cx="14" cy="14" r="10" stroke="white" strokeWidth="2" />
              <path
                d="M14 8C14 8 10 11.5 10 14.5C10 16.5 11.8 18 14 18S18 16.5 18 14.5C18 11.5 14 8 14 8Z"
                fill="white"
                opacity=".9"
              />
            </svg>
          </div>
          <div className="font-display text-2xl font-bold text-ink mb-1">
            Welcome to Serenova
          </div>
          <div className="text-[13px] text-muted">Sign in to your account</div>
        </div>

        <div className="p-7">
          <div className="mb-4">
            <div className="text-[12px] font-semibold text-mid mb-1.5">Email</div>
            <input
              defaultValue="priya@serenova.health"
              className="w-full px-4 py-3 border border-border rounded-xl text-[14px] font-body outline-none text-ink focus:border-rose transition-colors"
            />
          </div>
          <div className="mb-5">
            <div className="text-[12px] font-semibold text-mid mb-1.5">Password</div>
            <input
              type="password"
              defaultValue="••••••••"
              className="w-full px-4 py-3 border border-border rounded-xl text-[14px] font-body outline-none text-ink focus:border-rose transition-colors"
            />
          </div>
          <button
            onClick={onLogin}
            className="w-full py-3.5 rounded-full bg-rose text-white font-semibold text-[15px] hover:bg-rose-dark transition-colors"
          >
            Sign in →
          </button>
          <button
            onClick={onClose}
            className="w-full mt-3 py-2 text-[12px] text-muted hover:text-ink transition-colors"
          >
            ← Back to website
          </button>
        </div>
      </div>
    </div>
  );
}
