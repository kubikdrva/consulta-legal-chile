"use client";

import { useState, useRef } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const MAX_QUERIES = 5;

const EXAMPLES = [
  "Cuantos dias de vacaciones tiene un trabajador?",
  "Que proteccion tiene una trabajadora embarazada?",
  "Cual es el plazo para devolver un producto comprado online?",
  "Que dice la ley de datos personales sobre mis derechos?",
];

type Source = {
  text: string;
  source: string;
};

type QueryResult = {
  answer: string;
  sources: Source[];
  grounded: boolean;
  retries: number;
};

export default function Home() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [queryCount, setQueryCount] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const remaining = MAX_QUERIES - queryCount;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim() || loading || remaining <= 0) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query }),
      });

      if (!res.ok) throw new Error(`Error ${res.status}`);

      const data = await res.json();
      setResult(data);
      setQueryCount((c) => c + 1);
    } catch {
      setError("No se pudo conectar con el servidor. Verifica que el backend este activo.");
    } finally {
      setLoading(false);
    }
  }

  function handleExample(q: string) {
    setQuery(q);
    inputRef.current?.focus();
  }

  return (
    <main className="flex-1 flex flex-col items-center px-4 pt-20 pb-16">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="mb-12">
          <h1
            className="text-2xl font-medium tracking-tight"
            style={{ color: "var(--fg)" }}
          >
            Consulta Legal Chile
          </h1>
          <p
            className="mt-2 text-sm leading-relaxed"
            style={{ color: "var(--muted)" }}
          >
            Consulta la legislacion chilena en lenguaje natural. Codigo del
            Trabajo, proteccion al consumidor, datos personales y migracion.
          </p>
        </div>

        {/* Search */}
        <form onSubmit={handleSubmit} className="relative">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Escribe tu pregunta legal..."
            disabled={remaining <= 0}
            className="w-full h-12 px-4 pr-24 rounded-lg text-sm outline-none transition-shadow duration-150"
            style={{
              background: "var(--surface)",
              color: "var(--fg)",
              border: "1px solid var(--border)",
              boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
            }}
            onFocus={(e) =>
              (e.currentTarget.style.boxShadow =
                "0 0 0 2px var(--fg), 0 1px 2px rgba(0,0,0,0.04)")
            }
            onBlur={(e) =>
              (e.currentTarget.style.boxShadow =
                "0 1px 2px rgba(0,0,0,0.04)")
            }
          />
          <button
            type="submit"
            disabled={!query.trim() || loading || remaining <= 0}
            className="absolute right-2 top-1/2 -translate-y-1/2 h-8 px-3 rounded-md text-xs font-medium transition-all duration-150 disabled:opacity-30"
            style={{
              background: "var(--accent)",
              color: "var(--bg)",
            }}
            onMouseDown={(e) =>
              (e.currentTarget.style.transform =
                "translateY(-50%) scale(0.97)")
            }
            onMouseUp={(e) =>
              (e.currentTarget.style.transform = "translateY(-50%)")
            }
            onMouseLeave={(e) =>
              (e.currentTarget.style.transform = "translateY(-50%)")
            }
          >
            {loading ? "..." : "Buscar"}
          </button>
        </form>

        {/* Rate limit */}
        <div
          className="mt-3 text-xs"
          style={{ color: "var(--muted)" }}
        >
          {remaining > 0
            ? `${remaining} consulta${remaining !== 1 ? "s" : ""} restante${remaining !== 1 ? "s" : ""} en esta sesion`
            : "Limite de consultas alcanzado para esta sesion de demo"}
        </div>

        {/* Examples */}
        {!result && !loading && (
          <div className="mt-8">
            <p
              className="text-xs font-medium mb-3"
              style={{ color: "var(--muted)" }}
            >
              Prueba con:
            </p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => handleExample(ex)}
                  className="text-xs px-3 py-1.5 rounded-full transition-colors duration-150"
                  style={{
                    background: "var(--source-bg)",
                    color: "var(--muted)",
                    border: "1px solid var(--border)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background =
                      "var(--surface-hover)";
                    e.currentTarget.style.color = "var(--fg)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background =
                      "var(--source-bg)";
                    e.currentTarget.style.color = "var(--muted)";
                  }}
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="mt-12 flex items-center gap-3">
            <div
              className="w-4 h-4 rounded-full border-2 border-t-transparent animate-spin"
              style={{ borderColor: "var(--border)", borderTopColor: "transparent" }}
            />
            <span className="text-sm" style={{ color: "var(--muted)" }}>
              Buscando en la legislacion...
            </span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div
            className="mt-8 text-sm px-4 py-3 rounded-lg"
            style={{
              background: "var(--source-bg)",
              color: "var(--fg)",
              border: "1px solid var(--border)",
            }}
          >
            {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="mt-10">
            {/* Answer */}
            <div
              className="text-sm leading-7"
              style={{ color: "var(--fg)" }}
            >
              {result.answer}
            </div>

            {/* Metadata */}
            <div
              className="mt-6 flex items-center gap-4 text-xs"
              style={{ color: "var(--muted)" }}
            >
              <span
                className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full"
                style={{
                  background: result.grounded
                    ? "rgba(34,197,94,0.1)"
                    : "rgba(234,179,8,0.1)",
                  color: result.grounded ? "#16a34a" : "#ca8a04",
                }}
              >
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{
                    background: result.grounded ? "#16a34a" : "#ca8a04",
                  }}
                />
                {result.grounded ? "Fundamentado" : "Baja confianza"}
              </span>
              {result.retries > 0 && (
                <span>{result.retries} reintento{result.retries > 1 ? "s" : ""}</span>
              )}
            </div>

            {/* Sources */}
            {result.sources.length > 0 && (
              <div className="mt-8">
                <p
                  className="text-xs font-medium mb-3"
                  style={{ color: "var(--muted)" }}
                >
                  Fuentes ({result.sources.length})
                </p>
                <div className="space-y-2">
                  {result.sources.map((source, i) => (
                    <details
                      key={i}
                      className="group rounded-lg overflow-hidden"
                      style={{
                        background: "var(--source-bg)",
                        border: "1px solid var(--border)",
                      }}
                    >
                      <summary
                        className="px-4 py-2.5 text-xs cursor-pointer select-none flex items-center justify-between"
                        style={{ color: "var(--fg)" }}
                      >
                        <span className="font-medium">{source.source}</span>
                        <svg
                          className="w-3 h-3 transition-transform duration-150 group-open:rotate-180"
                          style={{ color: "var(--muted)" }}
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={2}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M19 9l-7 7-7-7"
                          />
                        </svg>
                      </summary>
                      <div
                        className="px-4 pb-3 text-xs leading-relaxed"
                        style={{
                          color: "var(--muted)",
                          fontFamily: "var(--font-geist-mono), monospace",
                        }}
                      >
                        {source.text}
                      </div>
                    </details>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <footer
        className="mt-auto pt-16 text-xs"
        style={{ color: "var(--muted)" }}
      >
        RAG Agent - Self-corrective retrieval over Chilean legislation
      </footer>
    </main>
  );
}
