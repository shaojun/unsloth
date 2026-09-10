// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * LLM-judged auto evaluation, decoupled from A/B tests: pick participant
 * sources directly, a judge model, and a prompt set — the run executes in the
 * background. Also hosts prompt set preview/editing and the judge report.
 */

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useT } from "@/i18n";
import {
  Add01Icon,
  Delete02Icon,
  EyeIcon,
  PencilEdit02Icon,
  RefreshIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import {
  type BattlegroundJudgeRun,
  type BattlegroundPrompt,
  type BattlegroundPromptSet,
  type BattlegroundSource,
  battlegroundApi,
} from "./api/battleground-api";
import { JudgeReportSection } from "./report-sections";

export function AutoEvalPanel() {
  const t = useT();
  const [promptSets, setPromptSets] = useState<BattlegroundPromptSet[]>([]);
  const [sources, setSources] = useState<BattlegroundSource[]>([]);
  const [runs, setRuns] = useState<BattlegroundJudgeRun[]>([]);
  const [createOpen, setCreateOpen] = useState(false);
  /** Prompt set currently previewed / edited. null = closed; editing flag distinguishes dialogs. */
  const [previewSet, setPreviewSet] = useState<BattlegroundPromptSet | null>(
    null,
  );
  const [editSet, setEditSet] = useState<BattlegroundPromptSet | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [sets, sourceList, runList] = await Promise.all([
        battlegroundApi.listPromptSets(),
        battlegroundApi.listSources(),
        battlegroundApi.listJudgeRuns(),
      ]);
      setPromptSets(sets);
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
    const active = runs.some(
      (run) => run.status === "running" || run.status === "queued",
    );
    if (!active) return;
    const timer = setInterval(() => void refresh(), 2500);
    return () => clearInterval(timer);
  }, [refresh, runs]);

  // Statuses-only signature: changes exactly when a run transitions (or a new
  // run appears), so the report section reloads when a background run lands.
  const runStatusKey = runs.map((run) => run.status).join(",");

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 p-4 sm:p-6">
      <section className="flex flex-col gap-3">
        <h2 className="font-heading font-semibold text-lg">
          {t("battleground.autoEvalTitle")}
        </h2>
        <p className="text-muted-foreground text-sm">
          {t("battleground.autoEvalDescription")}
        </p>
        <NewJudgeRunForm
          sources={sources}
          promptSets={promptSets}
          onStarted={refresh}
        />
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="font-heading font-semibold text-lg">
          {t("battleground.judgeRuns")}
        </h2>
        {runs.length === 0 && (
          <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
            {t("battleground.noJudgeRuns")}
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
          <h2 className="font-heading font-semibold text-lg">
            {t("battleground.promptSets")}
          </h2>
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <HugeiconsIcon icon={Add01Icon} className="size-4" />
            {t("battleground.newPromptSet")}
          </Button>
        </div>
        {promptSets.length === 0 && (
          <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
            {t("battleground.noPromptSets")}
          </p>
        )}
        <div className="flex flex-col gap-2">
          {promptSets.map((set) => (
            <div
              key={set.id}
              className="flex items-center gap-3 rounded-lg border p-3"
            >
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium">{set.name}</div>
                <p className="text-muted-foreground truncate text-xs">
                  {set.description || t("battleground.noDescription")}
                </p>
              </div>
              <Badge variant="secondary">
                {set.prompt_count} {t("battleground.promptsCount")}
              </Badge>
              <Button
                variant="outline"
                size="sm"
                disabled={set.prompt_count === 0}
                onClick={() => setPreviewSet(set)}
              >
                <HugeiconsIcon icon={EyeIcon} className="size-3.5" />
                {t("battleground.preview")}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setEditSet(set)}
              >
                <HugeiconsIcon icon={PencilEdit02Icon} className="size-3.5" />
                {t("battleground.edit")}
              </Button>
              {set.origin === "manual" && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => void handleDeleteSet(set.id)}
                >
                  <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                </Button>
              )}
            </div>
          ))}
        </div>
      </section>

      <JudgeReportSection refreshKey={runStatusKey} />

      <NewPromptSetDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        onCreated={refresh}
      />
      <PromptSetPreviewDialog
        set={previewSet}
        onClose={() => setPreviewSet(null)}
      />
      <PromptSetEditDialog
        set={editSet}
        onClose={() => setEditSet(null)}
        onSaved={refresh}
      />
    </div>
  );

  async function handleDeleteSet(id: string) {
    try {
      await battlegroundApi.deletePromptSet(id);
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
  run: BattlegroundJudgeRun;
  onChanged: () => void;
}) {
  const t = useT();
  const busy = run.status === "running" || run.status === "queued";
  const pct =
    run.progress_total > 0
      ? Math.round((run.progress_done / run.progress_total) * 100)
      : 0;

  const handleCancel = async () => {
    try {
      await battlegroundApi.cancelJudgeRun(run.id);
      onChanged();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  };

  const participants =
    run.source_names && run.source_names.length > 0
      ? run.source_names.join(" vs ")
      : t("battleground.notReady");

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border p-3">
      <Badge
        variant={
          run.status === "done"
            ? "default"
            : run.status === "error"
              ? "destructive"
              : "secondary"
        }
      >
        {run.status}
      </Badge>
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium">
          {participants} ·{" "}
          {run.mode === "pairwise"
            ? t("battleground.modePairwise")
            : t("battleground.modeRubric")}
        </div>
        <p className="text-muted-foreground truncate text-xs">
          {t("battleground.judge")}:{" "}
          {run.judge_source_name || run.judge_source_id.slice(0, 14)} ·{" "}
          {run.progress_done}/{run.progress_total}
          {run.error ? ` · ${run.error}` : ""}
        </p>
        {busy && (
          <div className="bg-muted mt-1.5 h-1.5 w-full overflow-hidden rounded-full">
            <div
              className="bg-primary h-full rounded-full transition-all"
              style={{ width: `${pct}%` }}
            />
          </div>
        )}
      </div>
      {busy && (
        <Button variant="outline" size="sm" onClick={() => void handleCancel()}>
          {t("battleground.cancel")}
        </Button>
      )}
    </div>
  );
}

function NewJudgeRunForm({
  sources,
  promptSets,
  onStarted,
}: {
  sources: BattlegroundSource[];
  promptSets: BattlegroundPromptSet[];
  onStarted: () => void;
}) {
  const t = useT();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [selectedPromptSetId, setSelectedPromptSetId] = useState("");
  const [judgeId, setJudgeId] = useState("");
  const [mode, setMode] = useState<"pairwise" | "rubric">("pairwise");
  const [maxPrompts, setMaxPrompts] = useState("20");
  const [running, setRunning] = useState(false);

  // Derived defaults instead of setState-in-effect: fall back to the first
  // entry until the user picks one.
  const promptSetId = selectedPromptSetId || promptSets[0]?.id || "";
  const participantCount = selectedIds.length;
  const pairInvalid = mode === "pairwise" && participantCount !== 2;
  const selfJudge = judgeId !== "" && selectedIds.includes(judgeId);

  const toggle = (id: string, checked: boolean) => {
    setSelectedIds((prev) =>
      checked ? [...prev, id] : prev.filter((sid) => sid !== id),
    );
  };

  const handleRun = async () => {
    setRunning(true);
    try {
      await battlegroundApi.createJudgeRun({
        source_ids: selectedIds,
        prompt_set_id: promptSetId,
        judge_source_id: judgeId,
        mode,
        max_prompts: maxPrompts ? Number(maxPrompts) : null,
      });
      toast.success(t("battleground.judgeStarted"));
      onStarted();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="grid gap-3 rounded-xl border p-4 sm:grid-cols-2">
      <div className="flex flex-col gap-1.5 sm:col-span-2">
        <Label>{t("battleground.participants")}</Label>
        <div className="flex flex-col gap-2 rounded-lg border p-2">
          {sources.length === 0 && (
            <p className="text-muted-foreground text-sm">
              {t("battleground.noSourcesYet")}
            </p>
          )}
          {sources.map((source) => (
            <label key={source.id} className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={selectedIds.includes(source.id)}
                onCheckedChange={(checked) =>
                  toggle(source.id, checked === true)
                }
              />
              <span className="min-w-0 flex-1 truncate">{source.name}</span>
            </label>
          ))}
        </div>
        {pairInvalid && (
          <p className="text-xs text-amber-500">
            {t("battleground.pairNeedsTwo")}
          </p>
        )}
        {selfJudge && (
          <p className="text-xs text-amber-500">
            {t("battleground.selfJudge")}
          </p>
        )}
      </div>
      <div className="flex flex-col gap-1.5">
        <Label>{t("battleground.promptSet")}</Label>
        <Select value={promptSetId} onValueChange={setSelectedPromptSetId}>
          <SelectTrigger>
            <SelectValue placeholder={t("battleground.noPromptSets")} />
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
        <Label>{t("battleground.judgeSource")}</Label>
        <Select value={judgeId} onValueChange={setJudgeId}>
          <SelectTrigger>
            <SelectValue placeholder={t("battleground.selectJudge")} />
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
        <Label>{t("battleground.judgeMode")}</Label>
        <div className="flex items-center gap-4 pt-2">
          <label className="flex items-center gap-2 text-sm">
            <Switch
              checked={mode === "pairwise"}
              onCheckedChange={(checked) => checked && setMode("pairwise")}
            />
            {t("battleground.modePairwise")}
          </label>
          <label className="flex items-center gap-2 text-sm">
            <Switch
              checked={mode === "rubric"}
              onCheckedChange={(checked) => checked && setMode("rubric")}
            />
            {t("battleground.modeRubric")}
          </label>
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label>{t("battleground.maxPrompts")}</Label>
        <Input
          value={maxPrompts}
          onChange={(event) => setMaxPrompts(event.target.value)}
        />
      </div>
      <div className="flex items-end justify-end sm:col-span-2">
        <Button
          disabled={
            participantCount === 0 ||
            pairInvalid ||
            selfJudge ||
            !promptSetId ||
            !judgeId ||
            running
          }
          onClick={() => void handleRun()}
        >
          <HugeiconsIcon icon={RefreshIcon} className="size-4" />
          {t("battleground.runJudge")}
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
      await battlegroundApi.createPromptSet({ name, description, prompts });
      toast.success(t("battleground.promptSetCreated"));
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
          <DialogTitle>{t("battleground.newPromptSet")}</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label>{t("battleground.sourceName")}</Label>
            <Input
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{t("battleground.notes")}</Label>
            <Input
              value={description}
              onChange={(event) => setDescription(event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{t("battleground.promptsBulk")}</Label>
            <Textarea
              value={bulk}
              rows={8}
              placeholder={t("battleground.promptsBulkHint")}
              onChange={(event) => setBulk(event.target.value)}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("battleground.cancel")}
          </Button>
          <Button
            disabled={!name.trim() || saving}
            onClick={() => void handleSave()}
          >
            {t("battleground.save")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** Read-only prompt list for a set. */
function PromptSetPreviewDialog({
  set,
  onClose,
}: {
  set: BattlegroundPromptSet | null;
  onClose: () => void;
}) {
  const t = useT();
  const [prompts, setPrompts] = useState<BattlegroundPrompt[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!set) return;
    setLoading(true);
    void battlegroundApi
      .listPrompts(set.id)
      .then(setPrompts)
      .catch((error: unknown) =>
        toast.error(error instanceof Error ? error.message : String(error)),
      )
      .finally(() => setLoading(false));
  }, [set]);

  return (
    <Dialog open={set != null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {set ? t("battleground.previewTitle", { name: set.name }) : ""}
          </DialogTitle>
        </DialogHeader>
        {loading ? (
          <p className="text-muted-foreground text-sm">
            {t("battleground.loading")}
          </p>
        ) : (
          <ol className="flex max-h-[50vh] list-decimal flex-col gap-2 overflow-y-auto pl-5">
            {prompts.map((prompt) => (
              <li key={prompt.id} className="text-sm">
                <span className="whitespace-pre-wrap">{prompt.prompt}</span>
                {prompt.reference_answer && (
                  <span className="text-muted-foreground block text-xs">
                    ✓ {prompt.reference_answer}
                  </span>
                )}
              </li>
            ))}
          </ol>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            {t("battleground.cancel")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** Edit a prompt set's metadata and its prompts (text + reference answer). */
function PromptSetEditDialog({
  set,
  onClose,
  onSaved,
}: {
  set: BattlegroundPromptSet | null;
  onClose: () => void;
  onSaved: () => void;
}) {
  const t = useT();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [prompts, setPrompts] = useState<BattlegroundPrompt[]>([]);
  const [newPrompt, setNewPrompt] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!set) return;
    setName(set.name);
    setDescription(set.description ?? "");
    setNewPrompt("");
    void battlegroundApi
      .listPrompts(set.id)
      .then(setPrompts)
      .catch((error: unknown) =>
        toast.error(error instanceof Error ? error.message : String(error)),
      );
  }, [set]);

  if (!set) return null;

  const save = async () => {
    setSaving(true);
    try {
      await battlegroundApi.updatePromptSet(set.id, { name, description });
      await Promise.all(
        prompts.map((prompt) =>
          battlegroundApi.updatePrompt(set.id, prompt.id, {
            prompt: prompt.prompt,
            reference_answer: prompt.reference_answer,
          }),
        ),
      );
      if (newPrompt.trim()) {
        await battlegroundApi.addPrompt(set.id, { prompt: newPrompt.trim() });
      }
      toast.success(t("battleground.promptSetSaved"));
      onSaved();
      onClose();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  };

  const removePrompt = async (promptId: string) => {
    try {
      await battlegroundApi.deletePrompt(set.id, promptId);
      setPrompts((prev) => prev.filter((prompt) => prompt.id !== promptId));
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <Dialog open={set != null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{t("battleground.editPromptSet")}</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label>{t("battleground.sourceName")}</Label>
            <Input
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{t("battleground.notes")}</Label>
            <Input
              value={description}
              onChange={(event) => setDescription(event.target.value)}
            />
          </div>
          <div className="flex max-h-[40vh] flex-col gap-2 overflow-y-auto">
            {prompts.map((prompt, index) => (
              <div
                key={prompt.id}
                className="flex flex-col gap-1.5 rounded-lg border p-2"
              >
                <div className="flex items-start gap-2">
                  <Textarea
                    value={prompt.prompt}
                    rows={2}
                    className="min-h-9 flex-1 resize-y text-sm"
                    onChange={(event) =>
                      setPrompts((prev) =>
                        prev.map((p, i) =>
                          i === index
                            ? { ...p, prompt: event.target.value }
                            : p,
                        ),
                      )
                    }
                  />
                  <Button
                    variant="ghost"
                    size="sm"
                    className="shrink-0"
                    onClick={() => void removePrompt(prompt.id)}
                  >
                    <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                  </Button>
                </div>
                <Input
                  value={prompt.reference_answer ?? ""}
                  placeholder={t("battleground.referenceAnswer")}
                  className="h-8 text-xs"
                  onChange={(event) =>
                    setPrompts((prev) =>
                      prev.map((p, i) =>
                        i === index
                          ? {
                              ...p,
                              reference_answer: event.target.value || null,
                            }
                          : p,
                      ),
                    )
                  }
                />
              </div>
            ))}
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{t("battleground.addPrompt")}</Label>
            <Textarea
              value={newPrompt}
              rows={2}
              className="min-h-9 resize-y"
              onChange={(event) => setNewPrompt(event.target.value)}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            {t("battleground.cancel")}
          </Button>
          <Button disabled={!name.trim() || saving} onClick={() => void save()}>
            {t("battleground.save")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
