// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Download04Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { authFetch } from "@/features/auth";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useT } from "@/i18n";
import {
  type PlaygroundReport,
  type PlaygroundReportSource,
  type PlaygroundTest,
  playgroundApi,
} from "./api/playground-api";

function pct(value: number | null): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(0)}%`;
}

function ciText(ci: [number, number] | null): string {
  if (!ci) return "";
  return `±${Math.round(((ci[1] - ci[0]) / 2) * 100)}%`;
}

export function ReportsPanel() {
  const t = useT();
  const [tests, setTests] = useState<PlaygroundTest[]>([]);
  const [scope, setScope] = useState<string>("all");
  const [report, setReport] = useState<PlaygroundReport | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async (scopeId: string) => {
    setLoading(true);
    try {
      const [testList, data] = await Promise.all([
        playgroundApi.listTests(true),
        playgroundApi.reportOverview(scopeId === "all" ? undefined : scopeId),
      ]);
      setTests(testList);
      setReport(data);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Load through a promise continuation (setState never runs in the effect body).
    void Promise.resolve().then(() => refresh(scope));
  }, [refresh, scope]);

  const downloadExport = async (kind: "dpo" | "sft" | "failures" | "feedback.csv") => {
    const url = playgroundApi.exportUrl(
      kind,
      scope === "all" ? undefined : scope,
    );
    try {
      const response = await authFetch(url);
      if (!response.ok) throw new Error(`Export failed (${response.status})`);
      const blob = await response.blob();
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `playground_${kind === "feedback.csv" ? "feedback.csv" : kind}.jsonl`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(link.href);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-6 p-4 sm:p-6">
      <div className="flex flex-wrap items-center gap-3">
        <Select value={scope} onValueChange={setScope}>
          <SelectTrigger className="w-64">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">{t("playground.allTests")}</SelectItem>
            {tests.map((test) => (
              <SelectItem key={test.id} value={test.id}>
                {test.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <div className="flex flex-wrap gap-1.5">
          <Button variant="outline" size="sm" onClick={() => void downloadExport("dpo")}>
            <HugeiconsIcon icon={Download04Icon} className="size-3.5" />
            DPO
          </Button>
          <Button variant="outline" size="sm" onClick={() => void downloadExport("sft")}>
            <HugeiconsIcon icon={Download04Icon} className="size-3.5" />
            SFT
          </Button>
          <Button variant="outline" size="sm" onClick={() => void downloadExport("failures")}>
            <HugeiconsIcon icon={Download04Icon} className="size-3.5" />
            {t("playground.exportFailures")}
          </Button>
          <Button variant="outline" size="sm" onClick={() => void downloadExport("feedback.csv")}>
            <HugeiconsIcon icon={Download04Icon} className="size-3.5" />
            CSV
          </Button>
        </div>
      </div>

      {loading && <p className="text-muted-foreground text-sm">{t("playground.loading")}</p>}

      {report && (
        <>
          <p className="text-muted-foreground text-xs">
            {t("playground.reportScope", {
              count: report.scope.test_count,
              sessions: report.scope.sessions,
            })}
          </p>
          {report.per_source.length === 0 && (
            <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
              {t("playground.noData")}
            </p>
          )}
          <div className="grid gap-3 md:grid-cols-2">
            {report.per_source.map((source) => (
              <SourceReportCard key={source.source_id} source={source} />
            ))}
          </div>

          {report.pairwise.length > 0 && (
            <section className="flex flex-col gap-2">
              <h2 className="font-heading font-semibold text-lg">
                {t("playground.headToHead")}
              </h2>
              <div className="overflow-hidden rounded-lg border">
                <table className="w-full text-sm">
                  <thead className="bg-muted/50 text-muted-foreground text-xs">
                    <tr>
                      <th className="px-3 py-2 text-left font-medium">{t("playground.pair")}</th>
                      <th className="px-3 py-2 text-right font-medium">{t("playground.humanWins")}</th>
                      <th className="px-3 py-2 text-right font-medium">{t("playground.judgeWins")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.pairwise.map((pair, index) => (
                      <tr key={index} className="border-t">
                        <td className="px-3 py-2">
                          {pair.name_a} <span className="text-muted-foreground">vs</span> {pair.name_b}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-xs">
                          {pair.human.a_wins}–{pair.human.b_wins}
                          {pair.human.ties ? ` (${pair.human.ties}${t("playground.tiesShort")})` : ""}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-xs">
                          {pair.judge.a_wins}–{pair.judge.b_wins}
                          {pair.judge.ties ? ` (${pair.judge.ties}${t("playground.tiesShort")})` : ""}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}

function SourceReportCard({ source }: { source: PlaygroundReportSource }) {
  const t = useT();
  const rubric = Object.entries(source.judge.rubric_avg);
  return (
    <div className="flex flex-col gap-3 rounded-xl border p-4">
      <div>
        <h3 className="truncate text-sm font-semibold">{source.name}</h3>
        <p className="text-muted-foreground text-xs">
          {source.responses} {t("playground.responses")}
        </p>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <Stat
          label={`${t("playground.human")} ${t("playground.winRate")}`}
          value={pct(source.human.win_rate)}
          hint={`${source.human.wins}W–${source.human.losses}L · ${ciText(source.human.win_rate_ci)}`}
        />
        <Stat
          label={`${t("playground.judge")} ${t("playground.winRate")}`}
          value={pct(source.judge.win_rate)}
          hint={`${source.judge.wins}W–${source.judge.losses}L · ${ciText(source.judge.win_rate_ci)}`}
        />
        <Stat
          label={t("playground.satisfaction")}
          value={pct(source.ratings.satisfaction)}
          hint={`👍 ${source.ratings.good} · 👎 ${source.ratings.bad}`}
        />
        <Stat
          label={t("playground.latency")}
          value={source.latency_ms.p50 != null ? `${source.latency_ms.p50}ms` : "—"}
          hint={`p95 ${source.latency_ms.p95 ?? "—"}ms`}
        />
      </div>
      {rubric.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {rubric.map(([dimension, score]) => (
            <span
              key={dimension}
              className="rounded-full border px-2 py-0.5 text-[11px]"
            >
              {dimension.replace("_", " ")} {score}
            </span>
          ))}
        </div>
      )}
      {Object.keys(source.tags).length > 0 && (
        <div className="flex flex-wrap gap-1">
          {Object.entries(source.tags).map(([tag, count]) => (
            <span
              key={tag}
              className="rounded-full bg-muted px-2 py-0.5 text-[11px]"
            >
              {tag} ×{count}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="bg-muted/40 rounded-lg border p-2">
      <p className="text-muted-foreground truncate text-[11px]">{label}</p>
      <p className="text-base leading-tight font-semibold">{value}</p>
      {hint && <p className="text-muted-foreground text-[10px]">{hint}</p>}
    </div>
  );
}
