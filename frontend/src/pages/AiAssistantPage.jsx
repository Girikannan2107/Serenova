import React, { useState, useEffect, useRef } from "react";
import {
  StatusBadge,
  SectionHeader,
  Card,
  HighlightCard,
  TypingIndicator,
  ApiResponseViewer,
} from "../components/common/CommonComponents";
import { askSerenovaAi } from "../services/api";

function ChatBox({ mode, data, initialQ = "" }) {
  const [messages, setMessages] = useState([
    {
      role: "ai",
      text: `Hi! I'm Serenova AI. Ask me anything about your ${
        mode === "postpartum" ? "postpartum recovery" : "pregnancy"
      } — vitals, risk score, nutrition, baby development, or what any reading means.`,
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollTop = chatEndRef.current.scrollHeight;
    }
  }, [messages, loading]);

  useEffect(() => {
    if (initialQ) {
      setInput(initialQ);
    }
  }, [initialQ]);

  const handleSend = async (customText) => {
    const textToSend = (customText || input).trim();
    if (!textToSend || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: textToSend }]);
    setLoading(true);

    try {
      const reply = await askSerenovaAi({
        message: textToSend,
        data,
        mode,
      });
      setMessages((prev) => [...prev, { role: "ai", text: reply }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          text: "Connection issue. Please check your network and try again.",
        },
      ]);
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <div
        ref={chatEndRef}
        className="flex-1 overflow-y-auto flex flex-col gap-3 pr-1 mb-3"
        style={{ maxHeight: "320px" }}
      >
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`flex flex-col ${
              m.role === "user" ? "items-end" : "items-start"
            }`}
          >
            {m.role === "ai" && (
              <span className="text-[10px] font-semibold text-plum mb-1 ml-1">
                Serenova AI
              </span>
            )}
            <div
              className={`px-3.5 py-2.5 rounded-2xl text-[13px] leading-relaxed max-w-[85%] ${
                m.role === "user"
                  ? "bg-rose text-white rounded-br-sm"
                  : "bg-white border border-border text-ink rounded-bl-sm"
              }`}
            >
              {m.text}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex flex-col items-start">
            <span className="text-[10px] font-semibold text-plum mb-1 ml-1">
              Serenova AI
            </span>
            <TypingIndicator />
          </div>
        )}
      </div>

      <div className="flex gap-2 flex-shrink-0">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              handleSend();
            }
          }}
          placeholder="Ask anything about your pregnancy…"
          className="flex-1 px-4 py-2.5 rounded-full border border-border text-[13px] font-body outline-none bg-white text-ink focus:border-rose transition-colors"
        />
        <button
          onClick={() => handleSend()}
          disabled={loading || !input.trim()}
          className="w-10 h-10 rounded-full bg-rose text-white flex items-center justify-center flex-shrink-0 hover:bg-rose-dark transition-colors disabled:opacity-40"
        >
          <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
            <path d="M2 7.5l11-5.5-4.5 5.5 4.5 5.5-11-5.5z" fill="white" />
          </svg>
        </button>
      </div>
    </div>
  );
}

export default function AiAssistantPage({ data, mode }) {
  const [selectedQuestion, setSelectedQuestion] = useState("");

  const suggestedQuestions =
    mode === "postpartum"
      ? [
          "How do I know if my PPD score is concerning?",
          "Tips for managing sleep fragmentation with a newborn",
          "Is my lochia normal at day 14?",
          "How to increase milk supply?",
          "Signs of postpartum anaemia?",
        ]
      : [
          "Why is my HRV declining this week?",
          "What does a risk score of 32 mean?",
          "What should I eat for folate deficiency?",
          "Is 37°C heat dangerous at week 28?",
          "When should I prepare my birth plan?",
        ];

  return (
    <div className="space-y-5 animate-fade-up">
      <div className="grid gap-5 lg:grid-cols-5">
        {/* Left Column: Risk & Suggested Questions */}
        <div className="lg:col-span-2 space-y-4">
          <HighlightCard>
            <div className="flex items-start gap-3 mb-4">
              <div className="w-9 h-9 rounded-full bg-plum text-white text-sm flex items-center justify-center flex-shrink-0">
                ✨
              </div>
              <div className="flex-1">
                <div className="font-display text-[14px] font-semibold text-plum-dark">
                  Risk analysis
                </div>
                <div className="text-[10px] text-muted">
                  Live · CNN + vital drift
                </div>
              </div>
              <div className="text-right">
                <div className="font-display text-2xl font-bold text-plum">
                  {data.risk.risk_score}
                  <span className="text-sm">/100</span>
                </div>
                <div className="text-[10px] text-muted capitalize">
                  {data.risk.risk_level} risk
                </div>
              </div>
            </div>

            {data.risk.positives.map((item, idx) => (
              <div
                key={idx}
                className="flex gap-2 text-[12px] text-ink py-1.5 border-b border-white/50"
              >
                <span className="text-sage font-bold flex-shrink-0">✓</span>
                {item}
              </div>
            ))}
            {data.risk.watchful_items.map((item, idx) => (
              <div
                key={idx}
                className="flex gap-2 text-[12px] text-ink py-1.5"
              >
                <span className="text-amber font-bold flex-shrink-0">~</span>
                {item}
              </div>
            ))}

            <div className="mt-3 p-2.5 bg-white/60 rounded-xl text-[11px] text-muted">
              {data.risk.plain_language_summary}
            </div>
          </HighlightCard>

          <Card>
            <SectionHeader title="Suggested questions" />
            <div className="space-y-2">
              {suggestedQuestions.map((q, idx) => (
                <div
                  key={idx}
                  onClick={() => setSelectedQuestion(q)}
                  className="px-3 py-2.5 border border-border rounded-xl text-[12px] text-mid cursor-pointer hover:border-rose hover:bg-rose-light transition-all"
                >
                  {q}
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Right Column: Interactive Chat Box */}
        <div className="lg:col-span-3">
          <Card className="flex flex-col" style={{ minHeight: "500px" }}>
            <SectionHeader
              title="Ask Serenova AI"
              right={<StatusBadge status="low" label="Powered by Claude" />}
            />
            <div className="flex-1">
              <ChatBox
                key={selectedQuestion + mode}
                mode={mode}
                data={data}
                initialQ={selectedQuestion}
              />
            </div>
            <ApiResponseViewer
              data={{
                risk: data.risk,
                conversation: data.conversation,
              }}
            />
          </Card>
        </div>
      </div>
    </div>
  );
}
