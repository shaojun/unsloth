// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type PlaygroundSourceKind = "local_dir" | "hf_model" | "external_openai";

export interface PlaygroundSource {
  id: string;
  kind: PlaygroundSourceKind;
  name: string;
  ref: string;
  external_model: string | null;
  api_key_set: boolean;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

export interface PlaygroundInstance {
  id: string;
  source_id: string;
  source_name: string | null;
  name: string;
  status: "merging" | "loading" | "running" | "stopping" | "stopped" | "error";
  model_slug: string;
  vllm_args: Record<string, unknown>;
  gpu_memory_utilization: number | null;
  merged_dir: string | null;
  error: string | null;
  started_at: string | null;
  stopped_at: string | null;
  created_at: string;
}

export interface PlaygroundTestSlot {
  source_id: string;
  source_name: string;
  kind: PlaygroundSourceKind | null;
  ready: boolean;
}

export interface PlaygroundTest {
  id: string;
  name: string;
  status: "active" | "archived";
  show_model_cards: boolean;
  reveal_after_vote: boolean;
  system_prompt: string | null;
  sampling: Record<string, unknown>;
  slots: PlaygroundTestSlot[];
  created_at: string;
  updated_at: string;
  share_url: string;
}

export interface PlaygroundJudgeRun {
  id: string;
  test_id: string;
  test_name?: string;
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

export interface PlaygroundJudgeResult {
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

export interface PlaygroundPromptSet {
  id: string;
  name: string;
  description: string | null;
  origin: string;
  prompt_count: number;
  created_at: string;
}

export interface PlaygroundPrompt {
  id: string;
  set_id: string;
  prompt: string;
  reference_answer: string | null;
  tags: string[];
  order_idx: number;
}

export interface PlaygroundReportSource {
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

export interface PlaygroundReport {
  scope: { test_id: string | null; test_count: number; sessions: number };
  per_source: PlaygroundReportSource[];
  pairwise: {
    source_a: string;
    source_b: string;
    name_a: string;
    name_b: string;
    human: { a_wins: number; b_wins: number; ties: number };
    judge: { a_wins: number; b_wins: number; ties: number };
  }[];
}

export interface VllmAvailability {
  installed: boolean;
  version: string | null;
  cli_path: string | null;
  importable: boolean;
  reason: string | null;
}

export const FEEDBACK_TAGS = [
  "correct",
  "instruction-following",
  "helpful",
  "style",
  "hallucination",
  "wrong-answer",
  "refused",
  "unsafe",
  "too-verbose",
  "too-short",
  "formatting",
  "other",
] as const;

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
        if (typeof structured.message === "string") message = structured.message;
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

export const playgroundApi = {
  // Sources
  listSources: () => jsonFetch<PlaygroundSource[]>("/api/playground/sources"),
  createSource: (body: Record<string, unknown>) =>
    jsonFetch<PlaygroundSource>("/api/playground/sources", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  updateSource: (id: string, body: Record<string, unknown>) =>
    jsonFetch<PlaygroundSource>(`/api/playground/sources/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deleteSource: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/playground/sources/${id}`, {
      method: "DELETE",
    }),
  testSource: (id: string) =>
    jsonFetch<{ ok: boolean; models: string[]; error?: string }>(
      `/api/playground/sources/${id}/test`,
      { method: "POST" },
    ),

  // Instances
  vllmAvailability: () =>
    jsonFetch<VllmAvailability>("/api/playground/vllm/availability"),
  listInstances: () =>
    jsonFetch<PlaygroundInstance[]>("/api/playground/instances"),
  hostInstance: (body: Record<string, unknown>) =>
    jsonFetch<PlaygroundInstance>("/api/playground/instances", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  stopInstance: (id: string) =>
    jsonFetch<PlaygroundInstance>(`/api/playground/instances/${id}/stop`, {
      method: "POST",
    }),
  /** Stop every live playground-hosted instance (training contention gate). */
  stopAllInstances: async (): Promise<number> => {
    const instances = await jsonFetch<PlaygroundInstance[]>(
      "/api/playground/instances",
    );
    const active = instances.filter(
      (instance) =>
        instance.status === "running" ||
        instance.status === "loading" ||
        instance.status === "merging",
    );
    let stopped = 0;
    for (const instance of active) {
      try {
        await jsonFetch(`/api/playground/instances/${instance.id}/stop`, {
          method: "POST",
        });
        stopped += 1;
      } catch {
        /* best-effort: stop as many as possible */
      }
    }
    return stopped;
  },
  deleteInstance: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/playground/instances/${id}`, {
      method: "DELETE",
    }),
  instanceLogs: (id: string, tail = 200) =>
    jsonFetch<{ logs: string }>(
      `/api/playground/instances/${id}/logs?tail=${tail}`,
    ),

  // Play sessions
  createPlaySession: (sourceId: string) =>
    jsonFetch<{ session_id: string }>("/api/playground/play/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id: sourceId }),
    }),
  getPlaySession: (sessionId: string) =>
    jsonFetch<PlaygroundPlaySession>(
      `/api/playground/play/sessions/${sessionId}`,
    ),
  submitFeedback: (body: Record<string, unknown>) =>
    jsonFetch<{ ok: boolean; feedback_id: string; revealed?: Record<string, string> }>(
      "/api/playground/feedback",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),

  // Tests
  listTests: (includeArchived = false) =>
    jsonFetch<PlaygroundTest[]>(
      `/api/playground/tests${includeArchived ? "?include_archived=true" : ""}`,
    ),
  createTest: (body: Record<string, unknown>) =>
    jsonFetch<PlaygroundTest>("/api/playground/tests", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deleteTest: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/playground/tests/${id}`, {
      method: "DELETE",
    }),

  // Prompt sets
  listPromptSets: () =>
    jsonFetch<PlaygroundPromptSet[]>("/api/playground/prompt-sets"),
  createPromptSet: (body: Record<string, unknown>) =>
    jsonFetch<PlaygroundPromptSet>("/api/playground/prompt-sets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deletePromptSet: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/playground/prompt-sets/${id}`, {
      method: "DELETE",
    }),
  listPrompts: (setId: string) =>
    jsonFetch<PlaygroundPrompt[]>(`/api/playground/prompt-sets/${setId}/prompts`),
  addPrompt: (setId: string, body: Record<string, unknown>) =>
    jsonFetch<PlaygroundPrompt>(`/api/playground/prompt-sets/${setId}/prompts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  // Judge
  listJudgeRuns: (testId?: string) =>
    jsonFetch<PlaygroundJudgeRun[]>(
      `/api/playground/judge/runs${testId ? `?test_id=${testId}` : ""}`,
    ),
  createJudgeRun: (body: Record<string, unknown>) =>
    jsonFetch<PlaygroundJudgeRun>("/api/playground/judge/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  getJudgeRun: (id: string) =>
    jsonFetch<PlaygroundJudgeRun & { results: PlaygroundJudgeResult[] }>(
      `/api/playground/judge/runs/${id}`,
    ),
  cancelJudgeRun: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/playground/judge/runs/${id}/cancel`, {
      method: "POST",
    }),

  // Reports
  reportOverview: (testId?: string) =>
    jsonFetch<PlaygroundReport>(
      `/api/playground/reports/overview${testId ? `?test_id=${testId}` : ""}`,
    ),
  exportUrl: (kind: "dpo" | "sft" | "failures" | "feedback.csv", testId?: string) =>
    `/api/playground/reports/export/${kind}${testId ? `?test_id=${testId}` : ""}`,
};

export interface PlaygroundPlaySessionTurn {
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

export interface PlaygroundPlaySession {
  session_id: string;
  source: PlaygroundSource | null;
  turns: PlaygroundPlaySessionTurn[];
}
