// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Add01Icon, Delete02Icon, PlayIcon, RefreshIcon, StopCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
import type { TranslationKey } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  type PlaygroundInstance,
  type PlaygroundSource,
  type PlaygroundSourceKind,
  type VllmAvailability,
  playgroundApi,
} from "./api/playground-api";

const SOURCE_KINDS: {
  value: PlaygroundSourceKind;
  labelKey: TranslationKey;
}[] = [
  { value: "local_dir", labelKey: "playground.kindLocal" },
  { value: "hf_model", labelKey: "playground.kindHf" },
  { value: "external_openai", labelKey: "playground.kindExternal" },
];

const INSTANCE_STATUS_STYLES: Record<string, string> = {
  running: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
  loading: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  merging: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  stopping: "bg-muted text-muted-foreground",
  stopped: "bg-muted text-muted-foreground",
  error: "bg-red-500/15 text-red-600 dark:text-red-400",
};

export function ModelsPanel() {
  const t = useT();
  const [sources, setSources] = useState<PlaygroundSource[]>([]);
  const [instances, setInstances] = useState<PlaygroundInstance[]>([]);
  const [availability, setAvailability] = useState<VllmAvailability | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [hostTarget, setHostTarget] = useState<PlaygroundSource | null>(null);
  const [logsInstance, setLogsInstance] = useState<PlaygroundInstance | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [src, inst, avail] = await Promise.all([
        playgroundApi.listSources(),
        playgroundApi.listInstances(),
        playgroundApi.vllmAvailability(),
      ]);
      setSources(src);
      setInstances(inst);
      setAvailability(avail);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    // Load through a promise continuation (setState never runs in the effect body).
    void Promise.resolve().then(refresh);
    // Poll while any instance is loading/merging so the badge flips on its own.
    const active = instances.some(
      (instance) => instance.status === "loading" || instance.status === "merging",
    );
    if (!active) return;
    const timer = setInterval(() => void refresh(), 3000);
    return () => clearInterval(timer);
  }, [refresh, instances]);

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 p-4 sm:p-6">
      <section className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h2 className="font-heading font-semibold text-lg">{t("playground.sources")}</h2>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => void refresh()}>
              <HugeiconsIcon icon={RefreshIcon} className="size-4" />
            </Button>
            <Button size="sm" onClick={() => setAddOpen(true)}>
              <HugeiconsIcon icon={Add01Icon} className="size-4" />
              {t("playground.addSource")}
            </Button>
          </div>
        </div>
        {sources.length === 0 && (
          <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
            {t("playground.noSources")}
          </p>
        )}
        <div className="flex flex-col gap-2">
          {sources.map((source) => (
            <SourceCard
              key={source.id}
              source={source}
              hosted={instances.some(
                (instance) =>
                  instance.source_id === source.id && instance.status === "running",
              )}
              onChanged={refresh}
              onHost={() => setHostTarget(source)}
            />
          ))}
        </div>
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="font-heading font-semibold text-lg">
          {t("playground.hostedInstances")}
        </h2>
        {availability && !availability.installed && (
          <p className="text-muted-foreground rounded-lg border border-dashed p-4 text-sm">
            {t("playground.vllmNotInstalled")}: {availability.reason}
          </p>
        )}
        {instances.length === 0 && (
          <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
            {t("playground.noInstances")}
          </p>
        )}
        <div className="flex flex-col gap-2">
          {instances.map((instance) => (
            <InstanceCard
              key={instance.id}
              instance={instance}
              onStopped={refresh}
              onShowLogs={() => setLogsInstance(instance)}
            />
          ))}
        </div>
      </section>

      <AddSourceDialog open={addOpen} onOpenChange={setAddOpen} onCreated={refresh} />
      <HostDialog source={hostTarget} onOpenChange={(open) => !open && setHostTarget(null)} onHosted={refresh} />
      <LogsDialog instance={logsInstance} onOpenChange={(open) => !open && setLogsInstance(null)} />
    </div>
  );
}

function SourceCard({
  source,
  hosted,
  onChanged,
  onHost,
}: {
  source: PlaygroundSource;
  hosted: boolean;
  onChanged: () => void;
  onHost: () => void;
}) {
  const t = useT();
  const [testing, setTesting] = useState(false);

  const kindLabel =
    SOURCE_KINDS.find((k) => k.value === source.kind)?.labelKey ?? "playground.kindExternal";

  const handleTest = async () => {
    setTesting(true);
    try {
      const result = await playgroundApi.testSource(source.id);
      if (result.ok) {
        toast.success(
          t("playground.connectionOk", { count: result.models.length }),
        );
      } else {
        toast.error(result.error || t("playground.connectionFailed"));
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setTesting(false);
    }
  };

  const handleDelete = async () => {
    try {
      await playgroundApi.deleteSource(source.id);
      toast.success(t("playground.sourceDeleted"));
      onChanged();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border p-3">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate font-medium text-sm">{source.name}</span>
          {hosted && (
            <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs font-medium text-emerald-600 dark:text-emerald-400">
              {t("playground.statusRunning")}
            </span>
          )}
        </div>
        <p className="text-muted-foreground truncate text-xs">
          {t(kindLabel)} · {source.ref}
          {source.external_model ? ` · ${source.external_model}` : ""}
        </p>
      </div>
      <div className="flex items-center gap-1.5">
        {source.kind !== "external_openai" && (
          <Button variant="outline" size="sm" onClick={onHost}>
            <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
            {t("playground.host")}
          </Button>
        )}
        <Button variant="outline" size="sm" disabled={testing} onClick={() => void handleTest()}>
          {t("playground.test")}
        </Button>
        <Button variant="ghost" size="sm" onClick={() => void handleDelete()}>
          <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
        </Button>
      </div>
    </div>
  );
}

function InstanceCard({
  instance,
  onStopped,
  onShowLogs,
}: {
  instance: PlaygroundInstance;
  onStopped: () => void;
  onShowLogs: () => void;
}) {
  const t = useT();
  const [busy, setBusy] = useState(false);

  const handleStop = async () => {
    setBusy(true);
    try {
      await playgroundApi.stopInstance(instance.id);
      onStopped();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async () => {
    setBusy(true);
    try {
      await playgroundApi.deleteInstance(instance.id);
      onStopped();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border p-3">
      <span
        className={cn(
          "rounded-full px-2 py-0.5 text-xs font-medium capitalize",
          INSTANCE_STATUS_STYLES[instance.status] ?? "bg-muted text-muted-foreground",
        )}
      >
        {instance.status}
      </span>
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium">
          {instance.name}
          <span className="text-muted-foreground ml-2 font-normal text-xs">
            {instance.model_slug}
          </span>
        </div>
        {instance.error && (
          <p className="truncate text-xs text-red-500">{instance.error}</p>
        )}
        {instance.gpu_memory_utilization != null && instance.status === "running" && (
          <p className="text-muted-foreground text-xs">
            GPU {(instance.gpu_memory_utilization * 100).toFixed(0)}%
          </p>
        )}
      </div>
      <div className="flex items-center gap-1.5">
        <Button variant="outline" size="sm" onClick={onShowLogs}>
          {t("playground.logs")}
        </Button>
        {instance.status === "running" || instance.status === "loading" || instance.status === "merging" ? (
          <Button variant="outline" size="sm" disabled={busy} onClick={() => void handleStop()}>
            <HugeiconsIcon icon={StopCircleIcon} className="size-3.5" />
            {t("playground.stop")}
          </Button>
        ) : (
          <Button variant="ghost" size="sm" disabled={busy} onClick={() => void handleDelete()}>
            <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
          </Button>
        )}
      </div>
    </div>
  );
}

function AddSourceDialog({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated: () => void;
}) {
  const t = useT();
  const [kind, setKind] = useState<PlaygroundSourceKind>("external_openai");
  const [name, setName] = useState("");
  const [ref, setRef] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      await playgroundApi.createSource({
        kind,
        name,
        ref,
        external_model: kind === "external_openai" ? model || null : null,
        api_key: kind === "external_openai" ? apiKey || null : null,
      });
      toast.success(t("playground.sourceAdded"));
      onOpenChange(false);
      setName("");
      setRef("");
      setModel("");
      setApiKey("");
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
          <DialogTitle>{t("playground.addSource")}</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label>{t("playground.sourceKind")}</Label>
            <Select value={kind} onValueChange={(value) => setKind(value as PlaygroundSourceKind)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SOURCE_KINDS.map((entry) => (
                  <SelectItem key={entry.value} value={entry.value}>
                    {t(entry.labelKey)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{t("playground.sourceName")}</Label>
            <Input value={name} onChange={(event) => setName(event.target.value)} />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>
              {kind === "external_openai" ? t("playground.baseUrl") : t("playground.modelPath")}
            </Label>
            <Input
              value={ref}
              placeholder={
                kind === "external_openai"
                  ? "https://api.openai.com/v1"
                  : kind === "hf_model"
                    ? "Qwen/Qwen3-8B"
                    : "/path/to/model or training run name"
              }
              onChange={(event) => setRef(event.target.value)}
            />
          </div>
          {kind === "external_openai" && (
            <>
              <div className="flex flex-col gap-1.5">
                <Label>{t("playground.modelId")}</Label>
                <Input value={model} onChange={(event) => setModel(event.target.value)} />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>{t("playground.apiKeyOptional")}</Label>
                <Input
                  type="password"
                  value={apiKey}
                  onChange={(event) => setApiKey(event.target.value)}
                />
              </div>
            </>
          )}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("playground.cancel")}
          </Button>
          <Button disabled={!name.trim() || !ref.trim() || saving} onClick={() => void handleSave()}>
            {t("playground.save")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function HostDialog({
  source,
  onOpenChange,
  onHosted,
}: {
  source: PlaygroundSource | null;
  onOpenChange: (open: boolean) => void;
  onHosted: () => void;
}) {
  const t = useT();
  const [gpuMemory, setGpuMemory] = useState("0.45");
  const [maxModelLen, setMaxModelLen] = useState("");
  const [extraArgs, setExtraArgs] = useState("");
  const [saving, setSaving] = useState(false);

  const handleHost = async () => {
    if (!source) return;
    setSaving(true);
    try {
      await playgroundApi.hostInstance({
        source_id: source.id,
        vllm_args: {
          gpu_memory_utilization: Number(gpuMemory) || undefined,
          max_model_len: maxModelLen ? Number(maxModelLen) : undefined,
        },
        extra_args: extraArgs ? extraArgs.split(/\s+/) : undefined,
      });
      toast.success(t("playground.hostStarting"));
      onOpenChange(false);
      onHosted();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={source !== null} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {t("playground.host")} · {source?.name}
          </DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label>{t("playground.gpuMemory")}</Label>
            <Input value={gpuMemory} onChange={(event) => setGpuMemory(event.target.value)} />
            <p className="text-muted-foreground text-xs">{t("playground.gpuMemoryHint")}</p>
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{t("playground.maxModelLen")}</Label>
            <Input
              value={maxModelLen}
              placeholder="8192"
              onChange={(event) => setMaxModelLen(event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{t("playground.extraArgs")}</Label>
            <Input
              value={extraArgs}
              placeholder="--enable-prefix-caching --dtype bfloat16"
              onChange={(event) => setExtraArgs(event.target.value)}
            />
          </div>
          {source && (source.ref as string).includes("adapter") && (
            <p className="text-xs text-amber-500">{t("playground.mergeNotice")}</p>
          )}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("playground.cancel")}
          </Button>
          <Button disabled={saving} onClick={() => void handleHost()}>
            {t("playground.host")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function LogsDialog({
  instance,
  onOpenChange,
}: {
  instance: PlaygroundInstance | null;
  onOpenChange: (open: boolean) => void;
}) {
  const t = useT();
  const [logs, setLogs] = useState("");

  useEffect(() => {
    if (!instance) return;
    let cancelled = false;
    const load = async () => {
      try {
        const result = await playgroundApi.instanceLogs(instance.id);
        if (!cancelled) setLogs(result.logs || "");
      } catch {
        /* ignore */
      }
    };
    void load();
    const timer = setInterval(() => void load(), 4000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [instance]);

  return (
    <Dialog open={instance !== null} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {t("playground.logs")} · {instance?.name}
          </DialogTitle>
        </DialogHeader>
        <pre className="bg-muted max-h-96 overflow-auto rounded-lg p-3 text-xs leading-relaxed whitespace-pre-wrap">
          {logs || t("playground.noLogs")}
        </pre>
      </DialogContent>
    </Dialog>
  );
}
