// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Add01Icon, Copy01Icon, Delete02Icon, EyeIcon, ViewOffSlashIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import {
  type PlaygroundSource,
  type PlaygroundTest,
  playgroundApi,
} from "./api/playground-api";

export function TestsPanel() {
  const t = useT();
  const [tests, setTests] = useState<PlaygroundTest[]>([]);
  const [createOpen, setCreateOpen] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setTests(await playgroundApi.listTests());
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
      await playgroundApi.deleteTest(id);
      await refresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 p-4 sm:p-6">
      <section className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h2 className="font-heading font-semibold text-lg">{t("playground.testsTitle")}</h2>
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <HugeiconsIcon icon={Add01Icon} className="size-4" />
            {t("playground.newTest")}
          </Button>
        </div>
        <p className="text-muted-foreground text-sm">{t("playground.testsDescription")}</p>
        {tests.length === 0 && (
          <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
            {t("playground.noTestsYet")}
          </p>
        )}
        <div className="flex flex-col gap-2">
          {tests.map((test) => (
            <div key={test.id} className="flex flex-col gap-2 rounded-lg border p-3">
              <div className="flex items-center gap-2">
                <HugeiconsIcon
                  icon={test.show_model_cards ? EyeIcon : ViewOffSlashIcon}
                  className="size-4 shrink-0 text-muted-foreground"
                />
                <span className="min-w-0 flex-1 truncate text-sm font-medium">{test.name}</span>
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
                  <span key={slot.source_id} className="rounded-full border px-2 py-0.5">
                    {slot.source_name}
                    {!slot.ready && ` · ${t("playground.notReady")}`}
                  </span>
                ))}
              </div>
              <ShareLinkRow test={test} />
            </div>
          ))}
        </div>
      </section>

      <NewTestDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        onCreated={refresh}
      />
    </div>
  );
}

function ShareLinkRow({ test }: { test: PlaygroundTest }) {
  const t = useT();
  const url = `${typeof window === "undefined" ? "" : window.location.origin}${test.share_url}`;

  return (
    <div className="bg-muted/40 flex items-center gap-2 rounded-md border p-1.5">
      <Input readOnly value={url} className="h-7 border-0 bg-transparent font-mono text-xs" />
      <Button
        variant="outline"
        size="sm"
        className="h-7 shrink-0"
        onClick={() => {
          void copyToClipboard(url);
          toast.success(t("playground.linkCopied"));
        }}
      >
        <HugeiconsIcon icon={Copy01Icon} className="size-3.5" />
        {t("playground.copyLink")}
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
  const [sources, setSources] = useState<PlaygroundSource[]>([]);
  const [name, setName] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [blind, setBlind] = useState(true);
  const [reveal, setReveal] = useState(true);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!open) return;
    void playgroundApi
      .listSources()
      .then(setSources)
      .catch(() => undefined);
  }, [open]);

  const handleSave = async () => {
    setSaving(true);
    try {
      await playgroundApi.createTest({
        name,
        slots: selected.map((sourceId) => ({ source_id: sourceId })),
        show_model_cards: !blind,
        reveal_after_vote: reveal,
        system_prompt: systemPrompt || null,
      });
      toast.success(t("playground.testCreated"));
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
          <DialogTitle>{t("playground.newTest")}</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label>{t("playground.testName")}</Label>
            <Input value={name} onChange={(event) => setName(event.target.value)} />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{t("playground.models")}</Label>
            <div className="flex flex-col gap-2 rounded-lg border p-2">
              {sources.map((source) => (
                <label key={source.id} className="flex items-center gap-2 text-sm">
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
                  <span className="text-muted-foreground text-xs capitalize">{source.kind.replace("_", " ")}</span>
                </label>
              ))}
            </div>
          </div>
          <label className="flex items-center justify-between gap-2 text-sm">
            <span>{t("playground.blindMode")}</span>
            <Switch checked={blind} onCheckedChange={setBlind} />
          </label>
          {blind && (
            <label className="flex items-center justify-between gap-2 text-sm">
              <span>{t("playground.revealAfterVote")}</span>
              <Switch checked={reveal} onCheckedChange={setReveal} />
            </label>
          )}
          <div className="flex flex-col gap-1.5">
            <Label>{t("playground.systemPrompt")}</Label>
            <Textarea
              value={systemPrompt}
              rows={3}
              placeholder={t("playground.systemPromptHint")}
              onChange={(event) => setSystemPrompt(event.target.value)}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("playground.cancel")}
          </Button>
          <Button
            disabled={!name.trim() || selected.length < 1 || saving}
            onClick={() => void handleSave()}
          >
            {t("playground.save")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
