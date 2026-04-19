import type { HedgeOutput, PortfolioInput } from "@/types/hedgeos";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: JSON.stringify(body) });
}

export type InstrumentType = "options" | "futures" | "inverse_etfs" | "forwards" | "swaps";

export const api = {
  health: () => request<{ status: string }>("/health"),

  /** Full 12-layer pipeline — all instrument types */
  analyzePortfolio: (payload: PortfolioInput) =>
    post<HedgeOutput>("/portfolio/analyze", payload),

  /** Pipeline restricted to a single instrument type */
  analyzeInstrument: (type: InstrumentType, payload: PortfolioInput) => {
    const slug = type === "inverse_etfs" ? "inverse-etfs" : type;
    return post<HedgeOutput>(`/instruments/${slug}`, payload);
  },
};
