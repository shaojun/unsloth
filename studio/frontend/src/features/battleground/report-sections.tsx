// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Report sections embedded in the tabs that produce the underlying data:
 *
 * - ``AbReportSection``   — human votes/ratings from A/B test sessions.
 * - ``JudgeReportSection`` — LLM-judged auto-eval results (all judge runs).
 *
 * Both read the same overview payload; each renders the half that belongs to
 * its tab.
 */

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { authFetch } from "@/features/auth";
import { useT } from "@/i18n";
import { Download04Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import {
  type BattlegroundReport,
  type BattlegroundReportSource,
  type BattlegroundTest,
  battlegroundApi,
} from "./api/battleground-api";

function pct(value: number | null): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(0)}%`;
}

function ciText(ci: [number, number] | null): string {
  if (!ci) return "";
  return `±${Math.round(((ci[1] - ci[0]) / 2) * 100)}%`;
}

function useReport() {
  const [tests, setTests] = useState<BattlegroundTest[]>([]);
  const [report, setReport] = useState<BattlegroundReport | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async (scopeId: string | null) => {
    setLoading(true);
    try {
      const [testList, data] = await Promise.all([
        battlegroundApi.listTests(true),
        battlegroundApi.reportOverview(scopeId ?? undefined),
      ]);
      setTests(testList);
      setReport(data);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  }, []);

  return { tests, report, loading, load };
}

function ReportHeading() {
  const t = useT();
  return (
    <section className="flex flex-col gap-3">
      <h2 className="font-heading font-semibold text-lg">
        {t("battleground.reportTitle")}
      </h2>
    </section>
  );
}

function Stat({
  label,
  value,
  hint,
}: { label: string; value: string; hint?: string }) {
  return (
    <div className="bg-muted/40 rounded-lg border p-2">
      <p className="text-muted-foreground truncate text-[11px]">{label}</p>
      <p className="text-base leading-tight font-semibold">{value}</p>
      {hint && <p className="text-muted-foreground text-[10px]">{hint}</p>}
    </div>
  );
}

/** Human (A/B tester) stats for one source. */
function HumanSourceCard({ source }: { source: BattlegroundReportSource }) {
  const t = useT();
  return (
    <div className="flex flex-col gap-3 rounded-xl border p-4">
      <div>
        <h3 className="truncate text-sm font-semibold">{source.name}</h3>
        <p className="text-muted-foreground text-xs">
          {source.responses} {t("battleground.responses")}
        </p>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <Stat
          label={`${t("battleground.human")} ${t("battleground.winRate")}`}
          value={pct(source.human.win_rate)}
          hint={`${source.human.wins}W–${source.human.losses}L · ${ciText(source.human.win_rate_ci)}`}
        />
        <Stat
          label={t("battleground.satisfaction")}
          value={pct(source.ratings.satisfaction)}
          hint={`👍 ${source.ratings.good} · 👎 ${source.ratings.bad}`}
        />
        <Stat
          label={t("battleground.latency")}
          value={
            source.latency_ms.p50 != null ? `${source.latency_ms.p50}ms` : "—"
          }
          hint={`p95 ${source.latency_ms.p95 ?? "—"}ms`}
        />
        <Stat
          label={t("battleground.responses")}
          value={String(source.responses)}
          hint={source.errors ? `⚠ ${source.errors}` : undefined}
        />
      </div>
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

/** LLM-judge stats for one source (win rate + rubric averages). */
function JudgeSourceCard({ source }: { source: BattlegroundReportSource }) {
  const t = useT();
  const rubric = Object.entries(source.judge.rubric_avg);
  return (
    <div className="flex flex-col gap-3 rounded-xl border p-4">
      <div>
        <h3 className="truncate text-sm font-semibold">{source.name}</h3>
        <p className="text-muted-foreground text-xs">
          {source.judge.total} {t("battleground.judgements")}
        </p>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <Stat
          label={`${t("battleground.judge")} ${t("battleground.winRate")}`}
          value={pct(source.judge.win_rate)}
          hint={`${source.judge.wins}W–${source.judge.losses}L · ${ciText(source.judge.win_rate_ci)}`}
        />
        <Stat
          label={t("battleground.latency")}
          value={
            source.latency_ms.p50 != null ? `${source.latency_ms.p50}ms` : "—"
          }
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
    </div>
  );
}

interface HeadToHeadProps {
  report: BattlegroundReport;
  side: "human" | "judge";
}

function HeadToHeadTable({ report, side }: HeadToHeadProps) {
  const t = useT();
  const pairs = report.pairwise.filter((pair) => {
    const counts = side === "human" ? pair.human : pair.judge;
    return counts.a_wins + counts.b_wins + counts.ties > 0;
  });
  if (pairs.length === 0) return null;
  return (
    <section className="flex flex-col gap-2">
      <h2 className="font-heading font-semibold text-lg">
        {t("battleground.headToHead")}
      </h2>
      <div className="overflow-hidden rounded-lg border">
        <table className="w-full text-sm">
          <thead className="bg-muted/50 text-muted-foreground text-xs">
            <tr>
              <th className="px-3 py-2 text-left font-medium">
                {t("battleground.pair")}
              </th>
              <th className="px-3 py-2 text-right font-medium">
                {side === "human"
                  ? t("battleground.humanWins")
                  : t("battleground.judgeWins")}
              </th>
            </tr>
          </thead>
          <tbody>
            {pairs.map((pair) => {
              const counts = side === "human" ? pair.human : pair.judge;
              return (
                <tr
                  key={`${pair.source_a}:${pair.source_b}`}
                  className="border-t"
                >
                  <td className="px-3 py-2">
                    {pair.name_a}{" "}
                    <span className="text-muted-foreground">vs</span>{" "}
                    {pair.name_b}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-xs">
                    {counts.a_wins}–{counts.b_wins}
                    {counts.ties
                      ? ` (${counts.ties}${t("battleground.tiesShort")})`
                      : ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function AbReportSection() {
  const t = useT();
  const { tests, report, loading, load } = useReport();
  const [scope, setScope] = useState<string>("all");
  const [exportFormat, setExportFormat] = useState<"jsonl" | "zip">("jsonl");

  useEffect(() => {
    void load(scope === "all" ? null : scope);
  }, [load, scope]);

  const downloadExport = async (
    kind: "dpo" | "sft" | "failures" | "feedback.csv",
  ) => {
    const url = battlegroundApi.exportUrl(
      kind,
      scope === "all" ? undefined : scope,
      kind === "feedback.csv" ? undefined : exportFormat,
    );
    try {
      const response = await authFetch(url);
      if (!response.ok) throw new Error(`Export failed (${response.status})`);
      const blob = await response.blob();
      const link = document.createElement("a");
      const objectUrl = URL.createObjectURL(blob);
      link.href = objectUrl;
      link.download =
        response.headers
          .get("Content-Disposition")
          ?.match(/filename="?([^";]+)"?/i)?.[1] ??
        (kind === "feedback.csv"
          ? "battleground_feedback.csv"
          : `battleground_${kind}.jsonl`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      // Keep the blob alive until the browser has started the download.
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <ReportHeading />
      <div className="flex flex-wrap items-center gap-3">
        <Select value={scope} onValueChange={setScope}>
          <SelectTrigger className="w-64">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">{t("battleground.allTests")}</SelectItem>
            {tests.map((test) => (
              <SelectItem key={test.id} value={test.id}>
                {test.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select
          value={exportFormat}
          onValueChange={(value) => setExportFormat(value as "jsonl" | "zip")}
        >
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="jsonl">{t("battleground.exportJsonl")}</SelectItem>
            <SelectItem value="zip">{t("battleground.exportZip")}</SelectItem>
          </SelectContent>
        </Select>
        <div className="flex flex-wrap gap-1.5">
          <Button
            variant="outline"
            size="sm"
            onClick={() => void downloadExport("dpo")}
          >
            <HugeiconsIcon icon={Download04Icon} className="size-3.5" />
            DPO
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => void downloadExport("sft")}
          >
            <HugeiconsIcon icon={Download04Icon} className="size-3.5" />
            SFT
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => void downloadExport("failures")}
          >
            <HugeiconsIcon icon={Download04Icon} className="size-3.5" />
            {t("battleground.exportFailures")}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => void downloadExport("feedback.csv")}
          >
            <HugeiconsIcon icon={Download04Icon} className="size-3.5" />
            CSV
          </Button>
        </div>
      </div>

      {loading && (
        <p className="text-muted-foreground text-sm">
          {t("battleground.loading")}
        </p>
      )}

      {report && (
        <>
          <p className="text-muted-foreground text-xs">
            {t("battleground.reportScope", {
              count: report.scope.test_count,
              sessions: report.scope.sessions,
            })}
          </p>
          {report.per_source.length === 0 ? (
            <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
              {t("battleground.noData")}
            </p>
          ) : (
            <div className="grid gap-3 md:grid-cols-2">
              {report.per_source.map((source) => (
                <HumanSourceCard key={source.source_id} source={source} />
              ))}
            </div>
          )}
          <HeadToHeadTable report={report} side="human" />
        </>
      )}
    </div>
  );
}

export function JudgeReportSection({
  refreshKey = 0,
}: {
  /**
   * Reload the report whenever this changes. The auto-eval panel passes a
   * signature of its run statuses, so a background run finishing (or a new
   * run appearing) refreshes the numbers without a manual reload.
   */
  refreshKey?: number | string;
}) {
  const t = useT();
  const { report, loading, load } = useReport();

  useEffect(() => {
    void load(null);
  }, [load, refreshKey]);

  return (
    <div className="flex flex-col gap-4">
      <ReportHeading />
      {loading && (
        <p className="text-muted-foreground text-sm">
          {t("battleground.loading")}
        </p>
      )}
      {report && (
        <>
          {report.per_source.length === 0 ? (
            <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
              {t("battleground.noData")}
            </p>
          ) : (
            <div className="grid gap-3 md:grid-cols-2">
              {report.per_source.map((source) => (
                <JudgeSourceCard key={source.source_id} source={source} />
              ))}
            </div>
          )}
          <HeadToHeadTable report={report} side="judge" />
        </>
      )}
    </div>
  );
}
