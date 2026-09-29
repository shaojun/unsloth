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
                {test.harness?.enabled && (
                  <span className="rounded-full border px-2 py-0.5">
                    {t("battleground.agentHarnessBadge")}
                  </span>
                )}
                {test.harness?.web_search?.enabled && (
                  <span className="rounded-full border px-2 py-0.5">
                    {t("battleground.webSearch")} · {test.harness.web_search.provider === "ddgs" ? "DDGS" : "Brave"}
                  </span>
                )}
                {(test.harness?.skills?.length ?? 0) > 0 && (
                  <span className="rounded-full border px-2 py-0.5">
                    {t("battleground.skillsCount", { count: test.harness.skills?.length ?? 0 })}
                  </span>
                )}
                {(test.harness?.mcp_servers?.length ?? 0) > 0 && (
                  <span className="rounded-full border px-2 py-0.5">
                    {t("battleground.mcpCount", { count: test.harness.mcp_servers?.length ?? 0 })}
                  </span>
                )}
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
  const [webSearch, setWebSearch] = useState(true);
  const [webProvider, setWebProvider] = useState<"brave" | "ddgs">("brave");
  const [webApiKey, setWebApiKey] = useState("");
  const [webMaxResults, setWebMaxResults] = useState("5");
  const [webCountry, setWebCountry] = useState("");
  const [webLanguage, setWebLanguage] = useState("");
  const [skillsJson, setSkillsJson] = useState("[]");
  const [mcpJson, setMcpJson] = useState('{\n  "mcpServers": {}\n}');
  const [maxTurns, setMaxTurns] = useState("8");
  const [toolTimeout, setToolTimeout] = useState("20");
  const [runTimeout, setRunTimeout] = useState("120");
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
      const skills = JSON.parse(skillsJson) as unknown;
      const mcpConfig = JSON.parse(mcpJson) as unknown;
      if (!Array.isArray(skills)) throw new Error(t("battleground.skillsJsonArray"));
      if (!mcpConfig || typeof mcpConfig !== "object" || Array.isArray(mcpConfig)) {
        throw new Error(t("battleground.mcpJsonObject"));
      }
      await battlegroundApi.createTest({
        name,
        slots: selected.map((sourceId) => ({ source_id: sourceId })),
        show_model_cards: !blind,
        reveal_after_vote: reveal,
        system_prompt: systemPrompt || null,
        harness: {
          enabled: true,
          version: 1,
          skills,
          web_search: {
            enabled: webSearch,
            provider: webProvider,
            api_key: webProvider === "brave" ? webApiKey : null,
            max_results: Number(webMaxResults),
            country: webCountry.trim() || null,
            language: webLanguage.trim() || null,
          },
          mcp_config: mcpConfig,
          max_turns: Number(maxTurns),
          tool_timeout_seconds: Number(toolTimeout),
          run_timeout_seconds: Number(runTimeout),
        },
      });
      toast.success(t("battleground.testCreated"));
      onOpenChange(false);
      setName("");
      setSelected([]);
      setSystemPrompt("");
      setWebApiKey("");
      onCreated();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-2xl">
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
          <div className="rounded-lg border p-3">
            <div className="mb-3">
              <Label>{t("battleground.sharedHarness")}</Label>
              <p className="text-muted-foreground mt-1 text-xs">
                {t("battleground.sharedHarnessHint")}
              </p>
            </div>
            <div className="mb-3 flex items-center justify-between gap-2 text-sm">
              <span>{t("battleground.webSearch")}</span>
              <Switch
                aria-label={t("battleground.webSearch")}
                checked={webSearch}
                onCheckedChange={setWebSearch}
              />
            </div>
            {webSearch && (
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="flex flex-col gap-1.5">
                  <Label>{t("battleground.searchProvider")}</Label>
                  <select
                    className="border-input bg-background h-9 rounded-md border px-3 text-sm"
                    value={webProvider}
                    onChange={(event) => setWebProvider(event.target.value as "brave" | "ddgs")}
                  >
                    <option value="brave">Brave Search</option>
                    <option value="ddgs">DDGS ({t("battleground.keylessFallback")})</option>
                  </select>
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label>{t("battleground.maxResults")}</Label>
                  <Input type="number" min="1" max="10" value={webMaxResults} onChange={(event) => setWebMaxResults(event.target.value)} />
                </div>
                {webProvider === "brave" && (
                  <div className="flex flex-col gap-1.5 sm:col-span-2">
                    <Label>{t("battleground.braveApiKey")}</Label>
                    <Input type="password" value={webApiKey} onChange={(event) => setWebApiKey(event.target.value)} />
                  </div>
                )}
                <div className="flex flex-col gap-1.5">
                  <Label>{t("battleground.countryOptional")}</Label>
                  <Input maxLength={2} placeholder="US" value={webCountry} onChange={(event) => setWebCountry(event.target.value.toUpperCase())} />
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label>{t("battleground.languageOptional")}</Label>
                  <Input maxLength={8} placeholder="en" value={webLanguage} onChange={(event) => setWebLanguage(event.target.value)} />
                </div>
              </div>
            )}
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{t("battleground.skillsJson")}</Label>
            <p className="text-muted-foreground text-xs">{t("battleground.skillsJsonHint")}</p>
            <Textarea value={skillsJson} rows={4} className="font-mono text-xs" onChange={(event) => setSkillsJson(event.target.value)} />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{t("battleground.mcpJson")}</Label>
            <p className="text-muted-foreground text-xs">{t("battleground.mcpJsonHint")}</p>
            <Textarea value={mcpJson} rows={7} className="font-mono text-xs" onChange={(event) => setMcpJson(event.target.value)} />
          </div>
          <details className="rounded-lg border p-3">
            <summary className="cursor-pointer text-sm font-medium">{t("battleground.agentLimits")}</summary>
            <div className="mt-3 grid gap-3 sm:grid-cols-3">
              <div className="flex flex-col gap-1.5"><Label>{t("battleground.maxAgentTurns")}</Label><Input type="number" min="1" max="30" value={maxTurns} onChange={(event) => setMaxTurns(event.target.value)} /></div>
              <div className="flex flex-col gap-1.5"><Label>{t("battleground.toolTimeout")}</Label><Input type="number" min="1" max="120" value={toolTimeout} onChange={(event) => setToolTimeout(event.target.value)} /></div>
              <div className="flex flex-col gap-1.5"><Label>{t("battleground.runTimeout")}</Label><Input type="number" min="5" max="600" value={runTimeout} onChange={(event) => setRunTimeout(event.target.value)} /></div>
            </div>
          </details>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("battleground.cancel")}
          </Button>
          <Button
            disabled={!name.trim() || selected.length < 1 || saving || (webSearch && webProvider === "brave" && !webApiKey.trim())}
            onClick={() => void handleSave()}
          >
            {t("battleground.save")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
