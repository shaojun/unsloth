// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Add01Icon, Delete02Icon, RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useT } from "@/i18n";
import {
  type PlaygroundJudgeRun,
  type PlaygroundPromptSet,
  type PlaygroundSource,
  type PlaygroundTest,
  playgroundApi,
} from "./api/playground-api";

export function EvaluatePanel() {
  const t = useT();
  const [promptSets, setPromptSets] = useState<PlaygroundPromptSet[]>([]);
  const [tests, setTests] = useState<PlaygroundTest[]>([]);
  const [sources, setSources] = useState<PlaygroundSource[]>([]);
  const [runs, setRuns] = useState<PlaygroundJudgeRun[]>([]);
  const [createOpen, setCreateOpen] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [sets, testList, sourceList, runList] = await Promise.all([
        playgroundApi.listPromptSets(),
        playgroundApi.listTests(),
        playgroundApi.listSources(),
        playgroundApi.listJudgeRuns(),
      ]);
      setPromptSets(sets);
      setTests(testList);
      setSources(sourceList);
      setRuns(runList);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    // Load through a promise continuation (setState never runs in the effect body).
    void Promise.resolve().then(refresh);
  }, [refresh]);

  // Poll while a run is in flight.
  useEffect(() => {
    const active = runs.some((run) => run.status === "running" || run.status === "queued");
    if (!active) return;
    const timer = setInterval(() => void refresh(), 2500);
    return () => clearInterval(timer);
  }, [refresh, runs]);

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 p-4 sm:p-6">
      <section className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h2 className="font-heading font-semibold text-lg">{t("playground.judgeTitle")}</h2>
        </div>
        <p className="text-muted-foreground text-sm">{t("playground.judgeDescription")}</p>
        <NewJudgeRunForm tests={tests} sources={sources} promptSets={promptSets} onStarted={refresh} />
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="font-heading font-semibold text-lg">{t("playground.judgeRuns")}</h2>
        {runs.length === 0 && (
          <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
            {t("playground.noJudgeRuns")}
          </p>
        )}
        <div className="flex flex-col gap-2">
          {runs.map((run) => (
            <JudgeRunRow key={run.id} run={run} onChanged={refresh} />
          ))}
        </div>
      </section>

      <section className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h2 className="font-heading font-semibold text-lg">{t("playground.promptSets")}</h2>
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <HugeiconsIcon icon={Add01Icon} className="size-4" />
            {t("playground.newPromptSet")}
          </Button>
        </div>
        {promptSets.length === 0 && (
          <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
            {t("playground.noPromptSets")}
          </p>
        )}
        <div className="flex flex-col gap-2">
          {promptSets.map((set) => (
            <div key={set.id} className="flex items-center gap-3 rounded-lg border p-3">
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium">{set.name}</div>
                <p className="text-muted-foreground truncate text-xs">
                  {set.description || t("playground.noDescription")}
                </p>
              </div>
              <Badge variant="secondary">
                {set.prompt_count} {t("playground.promptsCount")}
              </Badge>
              {set.origin === "manual" && (
                <Button variant="ghost" size="sm" onClick={() => void handleDeleteSet(set.id)}>
                  <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                </Button>
              )}
            </div>
          ))}
        </div>
      </section>

      <NewPromptSetDialog open={createOpen} onOpenChange={setCreateOpen} onCreated={refresh} />
    </div>
  );

  async function handleDeleteSet(id: string) {
    try {
      await playgroundApi.deletePromptSet(id);
      await refresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  }
}

function JudgeRunRow({
  run,
  onChanged,
}: {
  run: PlaygroundJudgeRun;
  onChanged: () => void;
}) {
  const t = useT();
  const busy = run.status === "running" || run.status === "queued";
  const pct =
    run.progress_total > 0 ? Math.round((run.progress_done / run.progress_total) * 100) : 0;

  const handleCancel = async () => {
    try {
      await playgroundApi.cancelJudgeRun(run.id);
      onChanged();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border p-3">
      <Badge
        variant={run.status === "done" ? "default" : run.status === "error" ? "destructive" : "secondary"}
      >
        {run.status}
      </Badge>
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium">
          {run.test_name || run.test_id.slice(0, 14)} · {run.mode === "pairwise" ? t("playground.modePairwise") : t("playground.modeRubric")}
        </div>
        <p className="text-muted-foreground truncate text-xs">
          {t("playground.judge")}: {run.judge_source_name || run.judge_source_id.slice(0, 14)} ·{" "}
          {run.progress_done}/{run.progress_total}
          {run.error ? ` · ${run.error}` : ""}
        </p>
        {busy && (
          <div className="bg-muted mt-1.5 h-1.5 w-full overflow-hidden rounded-full">
            <div className="bg-primary h-full rounded-full transition-all" style={{ width: `${pct}%` }} />
          </div>
        )}
      </div>
      {busy && (
        <Button variant="outline" size="sm" onClick={() => void handleCancel()}>
          {t("playground.cancel")}
        </Button>
      )}
    </div>
  );
}

function NewJudgeRunForm({
  tests,
  sources,
  promptSets,
  onStarted,
}: {
  tests: PlaygroundTest[];
  sources: PlaygroundSource[];
  promptSets: PlaygroundPromptSet[];
  onStarted: () => void;
}) {
  const t = useT();
  const [selectedTestId, setSelectedTestId] = useState("");
  const [selectedPromptSetId, setSelectedPromptSetId] = useState("");
  const [judgeId, setJudgeId] = useState("");
  const [mode, setMode] = useState<"pairwise" | "rubric">("pairwise");
  const [maxPrompts, setMaxPrompts] = useState("20");
  const [running, setRunning] = useState(false);

  // Derived defaults instead of setState-in-effect: fall back to the first
  // entry until the user picks one.
  const testId = selectedTestId || tests[0]?.id || "";
  const promptSetId = selectedPromptSetId || promptSets[0]?.id || "";

  const handleRun = async () => {
    setRunning(true);
    try {
      await playgroundApi.createJudgeRun({
        test_id: testId,
        prompt_set_id: promptSetId,
        judge_source_id: judgeId,
        mode,
        max_prompts: maxPrompts ? Number(maxPrompts) : null,
      });
      toast.success(t("playground.judgeStarted"));
      onStarted();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="grid gap-3 rounded-xl border p-4 sm:grid-cols-2">
      <div className="flex flex-col gap-1.5">
        <Label>{t("playground.testName")}</Label>
        <Select value={testId} onValueChange={setSelectedTestId}>
          <SelectTrigger>
            <SelectValue placeholder={t("playground.noTestsYet")} />
          </SelectTrigger>
          <SelectContent>
            {tests.map((test) => (
              <SelectItem key={test.id} value={test.id}>
                {test.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label>{t("playground.promptSet")}</Label>
        <Select value={promptSetId} onValueChange={setSelectedPromptSetId}>
          <SelectTrigger>
            <SelectValue placeholder={t("playground.noPromptSets")} />
          </SelectTrigger>
          <SelectContent>
            {promptSets.map((set) => (
              <SelectItem key={set.id} value={set.id}>
                {set.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label>{t("playground.judgeSource")}</Label>
        <Select value={judgeId} onValueChange={setJudgeId}>
          <SelectTrigger>
            <SelectValue placeholder={t("playground.selectJudge")} />
          </SelectTrigger>
          <SelectContent>
            {sources.map((source) => (
              <SelectItem key={source.id} value={source.id}>
                {source.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label>{t("playground.judgeMode")}</Label>
        <div className="flex items-center gap-4 pt-2">
          <label className="flex items-center gap-2 text-sm">
            <Switch checked={mode === "pairwise"} onCheckedChange={(checked) => checked && setMode("pairwise")} />
            {t("playground.modePairwise")}
          </label>
          <label className="flex items-center gap-2 text-sm">
            <Switch checked={mode === "rubric"} onCheckedChange={(checked) => checked && setMode("rubric")} />
            {t("playground.modeRubric")}
          </label>
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label>{t("playground.maxPrompts")}</Label>
        <Input value={maxPrompts} onChange={(event) => setMaxPrompts(event.target.value)} />
      </div>
      <div className="flex items-end justify-end">
        <Button
          disabled={!testId || !promptSetId || !judgeId || running}
          onClick={() => void handleRun()}
        >
          <HugeiconsIcon icon={RefreshIcon} className="size-4" />
          {t("playground.runJudge")}
        </Button>
      </div>
    </div>
  );
}

function NewPromptSetDialog({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated: () => void;
}) {
  const t = useT();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [bulk, setBulk] = useState("");
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      const prompts = bulk
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
          const [prompt, reference] = line.split(" |");
          return {
            prompt: prompt.trim(),
            reference_answer: reference ? reference.trim() : null,
          };
        });
      await playgroundApi.createPromptSet({ name, description, prompts });
      toast.success(t("playground.promptSetCreated"));
      onOpenChange(false);
      setName("");
      setDescription("");
      setBulk("");
      onCreated();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{t("playground.newPromptSet")}</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label>{t("playground.sourceName")}</Label>
            <Input value={name} onChange={(event) => setName(event.target.value)} />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{t("playground.notes")}</Label>
            <Input value={description} onChange={(event) => setDescription(event.target.value)} />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{t("playground.promptsBulk")}</Label>
            <Textarea
              value={bulk}
              rows={8}
              placeholder={t("playground.promptsBulkHint")}
              onChange={(event) => setBulk(event.target.value)}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("playground.cancel")}
          </Button>
          <Button disabled={!name.trim() || saving} onClick={() => void handleSave()}>
            {t("playground.save")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
