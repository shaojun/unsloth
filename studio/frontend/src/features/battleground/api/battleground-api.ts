// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type BattlegroundSourceKind = "external_openai";

export interface BattlegroundSource {
  id: string;
  kind: BattlegroundSourceKind;
  name: string;
  ref: string;
  external_model: string | null;
  api_key_set: boolean;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

/** A successfully trained run, for the manual `vllm serve` hosting hints. */
export interface BattlegroundTrainedModel {
  run_id: string;
  name: string;
  model_name: string | null;
  output_dir: string;
  /** "model" = full weights (config.json); "adapter" = LoRA adapter dir; "missing" = unknown. */
  artifact: "model" | "adapter" | "missing";
  finished_at: string | null;
}

export interface BattlegroundTestSlot {
  source_id: string;
  source_name: string;
  kind: BattlegroundSourceKind | null;
  ready: boolean;
}

export interface BattlegroundTest {
  id: string;
  name: string;
  status: "active" | "archived";
  show_model_cards: boolean;
  reveal_after_vote: boolean;
  system_prompt: string | null;
  sampling: Record<string, unknown>;
  slots: BattlegroundTestSlot[];
  created_at: string;
  updated_at: string;
  share_url: string;
}

export interface BattlegroundJudgeRun {
  id: string;
  test_id: string | null;
  test_name?: string | null;
  /** Participant source ids (standalone runs; legacy runs fall back to test slots). */
  source_ids?: string[];
  source_names?: string[];
  prompt_set_id: string;
  judge_source_id: string;
  judge_source_name?: string;
  mode: "pairwise" | "rubric";
  status: "queued" | "running" | "done" | "error" | "cancelled";
  session_id: string | null;
  progress_done: number;
  progress_total: number;
  error: string | null;
  created_at: string;
  finished_at: string | null;
  config: Record<string, unknown>;
}

export interface BattlegroundJudgeResult {
  id: string;
  run_id: string;
  prompt_id: string;
  turn_id: string;
  source_a: string | null;
  source_b: string | null;
  winner: string | null;
  reason: string | null;
  confidence: number | null;
  rubric_scores: Record<string, number>;
  judge_meta: Record<string, unknown>;
}

export interface BattlegroundPromptSet {
  id: string;
  name: string;
  description: string | null;
  origin: string;
  prompt_count: number;
  created_at: string;
}

export interface BattlegroundPrompt {
  id: string;
  set_id: string;
  prompt: string;
  reference_answer: string | null;
  tags: string[];
  order_idx: number;
}

export interface BattlegroundReportSource {
  source_id: string;
  name: string;
  kind: string | null;
  human: {
    wins: number;
    losses: number;
    ties: number;
    good: number;
    bad: number;
    total: number;
    win_rate: number | null;
    win_rate_ci: [number, number] | null;
  };
  ratings: {
    good: number;
    bad: number;
    total: number;
    satisfaction: number | null;
    satisfaction_ci: [number, number] | null;
  };
  judge: {
    wins: number;
    losses: number;
    ties: number;
    total: number;
    win_rate: number | null;
    win_rate_ci: [number, number] | null;
    rubric_avg: Record<string, number>;
  };
  tags: Record<string, number>;
  latency_ms: { p50: number | null; p95: number | null; avg: number | null };
  responses: number;
  errors: number;
}

export interface BattlegroundReport {
  scope: { test_id: string | null; test_count: number; sessions: number };
  per_source: BattlegroundReportSource[];
  pairwise: {
    source_a: string;
    source_b: string;
    name_a: string;
    name_b: string;
    human: { a_wins: number; b_wins: number; ties: number };
    judge: { a_wins: number; b_wins: number; ties: number };
  }[];
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await authFetch(path, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") {
        message = body.detail;
      } else if (body.detail && typeof body.detail === "object") {
        const structured = body.detail as { message?: unknown };
        if (typeof structured.message === "string")
          message = structured.message;
      }
    } catch {
      /* fall through with generic message */
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function post(path: string, body?: unknown): Promise<unknown> {
  return jsonFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export function put(path: string, body: unknown): Promise<unknown> {
  return jsonFetch(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function del(path: string): Promise<unknown> {
  return jsonFetch(path, { method: "DELETE" });
}

export const battlegroundApi = {
  // Sources
  listSources: () =>
    jsonFetch<BattlegroundSource[]>("/api/battleground/sources"),
  createSource: (body: Record<string, unknown>) =>
    jsonFetch<BattlegroundSource>("/api/battleground/sources", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  updateSource: (id: string, body: Record<string, unknown>) =>
    jsonFetch<BattlegroundSource>(`/api/battleground/sources/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deleteSource: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/battleground/sources/${id}`, {
      method: "DELETE",
    }),
  testSource: (id: string) =>
    jsonFetch<{ ok: boolean; models: string[]; error?: string }>(
      `/api/battleground/sources/${id}/test`,
      { method: "POST" },
    ),

  // Manual hosting hints (successfully trained runs)
  trainedModels: (limit = 50) =>
    jsonFetch<{ models: BattlegroundTrainedModel[] }>(
      `/api/battleground/trained-models?limit=${limit}`,
    ),

  // Play sessions
  createPlaySession: (sourceId: string) =>
    jsonFetch<{ session_id: string }>("/api/battleground/play/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id: sourceId }),
    }),
  getPlaySession: (sessionId: string) =>
    jsonFetch<BattlegroundPlaySession>(
      `/api/battleground/play/sessions/${sessionId}`,
    ),

  // Tests
  listTests: (includeArchived = false) =>
    jsonFetch<BattlegroundTest[]>(
      `/api/battleground/tests${includeArchived ? "?include_archived=true" : ""}`,
    ),
  createTest: (body: Record<string, unknown>) =>
    jsonFetch<BattlegroundTest>("/api/battleground/tests", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  updateTest: (id: string, body: Record<string, unknown>) =>
    jsonFetch<BattlegroundTest>(`/api/battleground/tests/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deleteTest: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/battleground/tests/${id}`, {
      method: "DELETE",
    }),

  // Prompt sets
  listPromptSets: () =>
    jsonFetch<BattlegroundPromptSet[]>("/api/battleground/prompt-sets"),
  createPromptSet: (body: Record<string, unknown>) =>
    jsonFetch<BattlegroundPromptSet>("/api/battleground/prompt-sets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  updatePromptSet: (id: string, body: Record<string, unknown>) =>
    jsonFetch<BattlegroundPromptSet>(`/api/battleground/prompt-sets/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deletePromptSet: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/battleground/prompt-sets/${id}`, {
      method: "DELETE",
    }),
  listPrompts: (setId: string) =>
    jsonFetch<BattlegroundPrompt[]>(
      `/api/battleground/prompt-sets/${setId}/prompts`,
    ),
  addPrompt: (setId: string, body: Record<string, unknown>) =>
    jsonFetch<BattlegroundPrompt>(
      `/api/battleground/prompt-sets/${setId}/prompts`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),
  updatePrompt: (
    setId: string,
    promptId: string,
    body: Record<string, unknown>,
  ) =>
    jsonFetch<BattlegroundPrompt>(
      `/api/battleground/prompt-sets/${setId}/prompts/${promptId}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),
  deletePrompt: (setId: string, promptId: string) =>
    jsonFetch<{ ok: boolean }>(
      `/api/battleground/prompt-sets/${setId}/prompts/${promptId}`,
      { method: "DELETE" },
    ),

  // Judge
  listJudgeRuns: (testId?: string) =>
    jsonFetch<BattlegroundJudgeRun[]>(
      `/api/battleground/judge/runs${testId ? `?test_id=${testId}` : ""}`,
    ),
  createJudgeRun: (body: Record<string, unknown>) =>
    jsonFetch<BattlegroundJudgeRun>("/api/battleground/judge/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  getJudgeRun: (id: string) =>
    jsonFetch<BattlegroundJudgeRun & { results: BattlegroundJudgeResult[] }>(
      `/api/battleground/judge/runs/${id}`,
    ),
  cancelJudgeRun: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/battleground/judge/runs/${id}/cancel`, {
      method: "POST",
    }),

  // Reports
  reportOverview: (testId?: string) =>
    jsonFetch<BattlegroundReport>(
      `/api/battleground/reports/overview${testId ? `?test_id=${testId}` : ""}`,
    ),
  exportUrl: (
    kind: "dpo" | "sft" | "failures" | "feedback.csv",
    testId?: string,
    format?: "jsonl" | "zip",
  ) => {
    const query = new URLSearchParams();
    if (testId) query.set("test_id", testId);
    if (format && kind !== "feedback.csv") query.set("format", format);
    const suffix = query.size ? `?${query.toString()}` : "";
    return `/api/battleground/reports/export/${kind}${suffix}`;
  },
};

export interface BattlegroundPlaySessionTurn {
  id: string;
  prompt: string;
  params: Record<string, unknown>;
  revealed: boolean;
  responses: {
    id: string;
    content: string;
    error: string | null;
    latency_ms: number | null;
    feedback: { rating: string; tags: string[]; comment: string | null } | null;
  }[];
}

export interface BattlegroundPlaySession {
  session_id: string;
  source: BattlegroundSource | null;
  turns: BattlegroundPlaySessionTurn[];
}
