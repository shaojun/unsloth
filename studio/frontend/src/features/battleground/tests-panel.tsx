// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import {
  Add01Icon,
  Copy01Icon,
  Delete02Icon,
  EyeIcon,
  ViewOffSlashIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import {
  type BattlegroundSource,
  type BattlegroundTest,
  battlegroundApi,
} from "./api/battleground-api";
import { AbReportSection } from "./report-sections";

export function TestsPanel() {
  const t = useT();
  const [tests, setTests] = useState<BattlegroundTest[]>([]);
  const [createOpen, setCreateOpen] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setTests(await battlegroundApi.listTests(true));
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    // Load through a promise continuation (setState never runs in the effect body).
    void Promise.resolve().then(refresh);
  }, [refresh]);

  const handleDelete = async (id: string) => {
    try {
      await battlegroundApi.deleteTest(id);
      await refresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  };

  const handleStop = async (id: string) => {
    if (!window.confirm(t("battleground.stopTestConfirm"))) return;
    try {
      await battlegroundApi.updateTest(id, { status: "archived" });
      toast.success(t("battleground.testStopped"));
      await refresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 p-4 sm:p-6">
      <section className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h2 className="font-heading font-semibold text-lg">
            {t("battleground.testsTitle")}
          </h2>
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <HugeiconsIcon icon={Add01Icon} className="size-4" />
            {t("battleground.newTest")}
          </Button>
        </div>
        <p className="text-muted-foreground text-sm">
          {t("battleground.testsDescription")}
        </p>
        {tests.length === 0 && (
          <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
            {t("battleground.noTestsYet")}
          </p>
        )}
        <div className="flex flex-col gap-2">
          {tests.map((test) => (
            <div
              key={test.id}
              className="flex flex-col gap-2 rounded-lg border p-3"
            >
              <div className="flex items-center gap-2">
                <HugeiconsIcon
                  icon={test.show_model_cards ? EyeIcon : ViewOffSlashIcon}
                  className="size-4 shrink-0 text-muted-foreground"
                />
                <span className="min-w-0 flex-1 truncate text-sm font-medium">
                  {test.name}
                </span>
                <span className={cn(
                  "rounded-full px-2 py-0.5 text-[11px]",
                  test.status === "active"
                    ? "bg-primary/10 text-primary"
                    : "bg-muted text-muted-foreground",
                )}>
                  {test.status === "active"
                    ? t("battleground.statusActive")
                    : t("battleground.statusStopped")}
                </span>
                {test.status === "active" && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => void handleStop(test.id)}
                  >
                    {t("battleground.stopTest")}
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => void handleDelete(test.id)}
                >
                  <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                </Button>
              </div>
              <div className="text-muted-foreground flex flex-wrap gap-1.5 text-xs">
                {test.slots.map((slot) => (
                  <span
                    key={slot.source_id}
                    className="rounded-full border px-2 py-0.5"
                  >
                    {slot.source_name}
                    {!slot.ready && ` · ${t("battleground.notReady")}`}
                  </span>
                ))}
              </div>
              <ShareLinkRow test={test} />
            </div>
          ))}
        </div>
      </section>

      <AbReportSection />

      <NewTestDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        onCreated={refresh}
      />
    </div>
  );
}

function ShareLinkRow({ test }: { test: BattlegroundTest }) {
  const t = useT();
  const url = `${typeof window === "undefined" ? "" : window.location.origin}${test.share_url}`;

  return (
    <div className="bg-muted/40 flex items-center gap-2 rounded-md border p-1.5">
      <Input
        readOnly={true}
        value={url}
        className="h-7 border-0 bg-transparent font-mono text-xs"
      />
      <Button
        variant="outline"
        size="sm"
        className="h-7 shrink-0"
        onClick={() => {
          void copyToClipboard(url);
          toast.success(t("battleground.linkCopied"));
        }}
      >
        <HugeiconsIcon icon={Copy01Icon} className="size-3.5" />
        {t("battleground.copyLink")}
      </Button>
    </div>
  );
}

function NewTestDialog({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated: () => void;
}) {
  const t = useT();
  const [sources, setSources] = useState<BattlegroundSource[]>([]);
  const [name, setName] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [blind, setBlind] = useState(true);
  const [reveal, setReveal] = useState(true);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!open) return;
    void battlegroundApi
      .listSources()
      .then(setSources)
      .catch(() => undefined);
  }, [open]);

  const handleSave = async () => {
    setSaving(true);
    try {
      await battlegroundApi.createTest({
        name,
        slots: selected.map((sourceId) => ({ source_id: sourceId })),
        show_model_cards: !blind,
        reveal_after_vote: reveal,
        system_prompt: systemPrompt || null,
      });
      toast.success(t("battleground.testCreated"));
      onOpenChange(false);
      setName("");
      setSelected([]);
      setSystemPrompt("");
      onCreated();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{t("battleground.newTest")}</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label>{t("battleground.testName")}</Label>
            <Input
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{t("battleground.models")}</Label>
            <div className="flex flex-col gap-2 rounded-lg border p-2">
              {sources.map((source) => (
                <label
                  key={source.id}
                  className="flex items-center gap-2 text-sm"
                >
                  <Checkbox
                    checked={selected.includes(source.id)}
                    onCheckedChange={(checked) =>
                      setSelected((prev) =>
                        checked
                          ? [...prev, source.id]
                          : prev.filter((id) => id !== source.id),
                      )
                    }
                  />
                  <span className="min-w-0 flex-1 truncate">{source.name}</span>
                  <span className="text-muted-foreground text-xs capitalize">
                    {source.kind.replace("_", " ")}
                  </span>
                </label>
              ))}
            </div>
          </div>
          <label className="flex items-center justify-between gap-2 text-sm">
            <span>{t("battleground.blindMode")}</span>
            <Switch checked={blind} onCheckedChange={setBlind} />
          </label>
          {blind && (
            <label className="flex items-center justify-between gap-2 text-sm">
              <span>{t("battleground.revealAfterVote")}</span>
              <Switch checked={reveal} onCheckedChange={setReveal} />
            </label>
          )}
          <div className="flex flex-col gap-1.5">
            <Label>{t("battleground.systemPrompt")}</Label>
            <Textarea
              value={systemPrompt}
              rows={3}
              placeholder={t("battleground.systemPromptHint")}
              onChange={(event) => setSystemPrompt(event.target.value)}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("battleground.cancel")}
          </Button>
          <Button
            disabled={!name.trim() || selected.length < 1 || saving}
            onClick={() => void handleSave()}
          >
            {t("battleground.save")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
