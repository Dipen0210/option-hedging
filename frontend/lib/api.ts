import type { HedgeOutput, PortfolioInput } from "@/types/hedgeos";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/**
 * Request timeouts, in milliseconds.
 *
 * The full pipeline fetches price history, option chains and rates for every
 * holding, then prices, sizes and scores each candidate. On a cold cache a
 * multi-holding portfolio takes well over a minute. Before these existed
 * `fetch` had no timeout at all, so a stalled upstream left the UI on
 * "Analyzing…" with no way out and no explanation — the user could not tell a
 * slow run from a dead one.
 *
 * The analyze budget is deliberately generous: cutting off a run that would
 * have succeeded is its own failure. It exists to bound the wait, not to
 * police it.
 */
const TIMEOUT_MS = {
  health: 5_000,
  analyze: 240_000,
} as const;

/** Why a request failed, for a UI that should say different things to each. */
export type ApiFailure =
  | "timeout"      // exceeded our own budget
  | "offline"      // request never reached the server
  | "validation"   // 422 — the portfolio itself was rejected
  | "auth"         // 401 / 403
  | "unavailable"  // 503 — a dependency is down or unconfigured
  | "server"       // 5xx
  | "client";      // other 4xx

export class ApiError extends Error {
  readonly kind: ApiFailure;
  readonly status?: number;
  /** Field-level messages from a 422, keyed by the field path. */
  readonly fieldErrors?: Record<string, string>;

  constructor(
    kind: ApiFailure,
    message: string,
    status?: number,
    fieldErrors?: Record<string, string>,
  ) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.fieldErrors = fieldErrors;
  }
}

/**
 * Turn a FastAPI error body into something a person can act on.
 *
 * The server already sanitises its own messages — `internal_error` returns a
 * context line plus a reference id and keeps the traceback in the log — so a
 * `detail` string is safe to show as-is and the reference id is worth showing,
 * because it is what support would ask for. What is NOT safe to show is the
 * raw response text for a non-JSON reply: an HTML error page from a proxy, or
 * a stack trace from a misconfigured deployment. Those are replaced.
 */
function describe(status: number, body: unknown, raw: string): ApiError {
  const detail = (body as { detail?: unknown } | null)?.detail;

  // 422 — Pydantic validation. `detail` is a list of per-field objects.
  if (status === 422 && Array.isArray(detail)) {
    const fieldErrors: Record<string, string> = {};
    for (const item of detail) {
      const e = item as { loc?: unknown[]; msg?: string };
      const path = Array.isArray(e.loc)
        ? e.loc.filter((p) => p !== "body").join(".")
        : "";
      if (e.msg) fieldErrors[path || "portfolio"] = e.msg;
    }
    const first = Object.entries(fieldErrors)[0];
    return new ApiError(
      "validation",
      first
        ? `${first[0]}: ${first[1]}`
        : "The portfolio was rejected as invalid.",
      status,
      fieldErrors,
    );
  }

  if (typeof detail === "string" && detail.trim()) {
    const kind: ApiFailure =
      status === 401 || status === 403
        ? "auth"
        : status === 503
          ? "unavailable"
          : status >= 500
            ? "server"
            : "client";
    return new ApiError(kind, detail, status);
  }

  // No usable JSON detail. Do not surface `raw` — it may be a proxy's HTML
  // error page or a traceback from a misconfigured server.
  const fallback =
    status >= 500
      ? "The analysis server reported an internal error. Nothing was charged or traded; try again shortly."
      : `The request was rejected (HTTP ${status}).`;
  if (raw && process.env.NODE_ENV !== "production") {
    console.error(`API ${status} — unparsed body:`, raw.slice(0, 2000));
  }
  return new ApiError(status >= 500 ? "server" : "client", fallback, status);
}

async function request<T>(
  path: string,
  options: RequestInit & { timeoutMs?: number } = {},
): Promise<T> {
  const { timeoutMs = TIMEOUT_MS.analyze, signal, ...init } = options;

  // AbortSignal.any lets a caller's own signal (an unmounting component, a
  // user pressing Cancel) coexist with the timeout instead of overriding it.
  const timeout = AbortSignal.timeout(timeoutMs);
  const combined = signal ? AbortSignal.any([signal, timeout]) : timeout;

  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
      signal: combined,
    });
  } catch (e) {
    if (timeout.aborted) {
      throw new ApiError(
        "timeout",
        `The analysis did not finish within ${Math.round(timeoutMs / 1000)}s. ` +
          `It may still be running on the server — no hedge was placed. ` +
          `Try again, or reduce the number of holdings.`,
      );
    }
    if (signal?.aborted) throw e;   // the caller cancelled; let them handle it
    throw new ApiError(
      "offline",
      `Could not reach the analysis server at ${API_URL}. ` +
        `Check that the backend is running and that NEXT_PUBLIC_API_URL points at it.`,
    );
  }

  const raw = await res.text();
  let body: unknown = null;
  if (raw) {
    try {
      body = JSON.parse(raw);
    } catch {
      /* handled by describe() */
    }
  }

  if (!res.ok) throw describe(res.status, body, raw);

  if (body === null) {
    throw new ApiError(
      "server",
      "The analysis server returned an empty response.",
      res.status,
    );
  }
  return body as T;
}

function post<T>(
  path: string,
  payload: unknown,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<T> {
  return request<T>(path, {
    method: "POST",
    body: JSON.stringify(payload),
    timeoutMs,
    signal,
  });
}

export type InstrumentType = "options" | "futures" | "inverse_etfs" | "forwards" | "swaps";

export const api = {
  health: (signal?: AbortSignal) =>
    request<{ status: string }>("/health", {
      timeoutMs: TIMEOUT_MS.health,
      signal,
    }),

  /** Full 12-layer pipeline — all instrument types */
  analyzePortfolio: (payload: PortfolioInput, signal?: AbortSignal) =>
    post<HedgeOutput>("/portfolio/analyze", payload, TIMEOUT_MS.analyze, signal),

  /** Pipeline restricted to a single instrument type */
  analyzeInstrument: (
    type: InstrumentType,
    payload: PortfolioInput,
    signal?: AbortSignal,
  ) => {
    const slug = type === "inverse_etfs" ? "inverse-etfs" : type;
    return post<HedgeOutput>(
      `/instruments/${slug}`,
      payload,
      TIMEOUT_MS.analyze,
      signal,
    );
  },
};
