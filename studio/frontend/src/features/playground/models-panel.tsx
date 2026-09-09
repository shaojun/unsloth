// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Add01Icon, Copy01Icon, Delete02Icon, RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
  type PlaygroundTrainedModel,
  playgroundApi,
} from "./api/playground-api";

/** Default port the hosting hints suggest for a manually started vLLM server. */
const VLLM_HINT_PORT = 8000;

export function ModelsPanel() {
  const t = useT();
  const [sources, setSources] = useState<PlaygroundSource[]>([]);
  const [trainedModels, setTrainedModels] = useState<PlaygroundTrainedModel[]>([]);
  const [addOpen, setAddOpen] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [src, trained] = await Promise.all([
        playgroundApi.listSources(),
        playgroundApi.trainedModels(),
      ]);
      setSources(src);
      setTrainedModels(trained.models);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    // Load through a promise continuation (setState never runs in the effect body).
    void Promise.resolve().then(refresh);
  }, [refresh]);

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 p-4 sm:p-6">
      <HostingHintSection models={trainedModels} />

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
            <SourceCard key={source.id} source={source} onChanged={refresh} />
          ))}
        </div>
      </section>

      <AddSourceDialog open={addOpen} onOpenChange={setAddOpen} onCreated={refresh} />
    </div>
  );
}

/**
 * The app never starts inference servers itself. This section tells users how
 * to host a trained model manually: copy a `vllm serve <path>` command (the
 * model's full output path is the key info) into a terminal on this machine,
 * then register the resulting endpoint as a source below.
 */
function HostingHintSection({ models }: { models: PlaygroundTrainedModel[] }) {
  const t = useT();

  return (
    <section className="flex flex-col gap-3">
      <h2 className="font-heading font-semibold text-lg">
        {t("playground.hostManuallyTitle")}
      </h2>
      <p className="text-muted-foreground text-sm">
        {t("playground.hostManuallyIntro")}
      </p>
      <ol className="text-muted-foreground flex list-decimal flex-col gap-1 pl-5 text-sm">
        <li>{t("playground.hostManuallyStep1")}</li>
        <li>{t("playground.hostManuallyStep2")}</li>
        <li>{t("playground.hostManuallyStep3")}</li>
      </ol>

      {models.length === 0 ? (
        <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
          {t("playground.noTrainedModels")}
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          {models.map((model) => (
            <TrainedModelRow key={model.run_id} model={model} />
          ))}
        </div>
      )}
    </section>
  );
}

function TrainedModelRow({ model }: { model: PlaygroundTrainedModel }) {
  const t = useT();
  const command = `vllm serve "${model.output_dir}" --port ${VLLM_HINT_PORT}`;

  return (
    <div className="flex flex-col gap-2 rounded-lg border p-3">
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-sm font-medium">{model.name}</span>
        {model.model_name && model.model_name !== model.name && (
          <span className="text-muted-foreground truncate text-xs">{model.model_name}</span>
        )}
      </div>
      {model.artifact === "adapter" && (
        <p className="text-xs text-amber-500">{t("playground.adapterNote")}</p>
      )}
      <div className="bg-muted/40 flex items-center gap-2 rounded-md border p-1.5">
        <code className="min-w-0 flex-1 overflow-x-auto font-mono text-xs whitespace-nowrap">
          {command}
        </code>
        <Button
          variant="outline"
          size="sm"
          className="h-7 shrink-0"
          onClick={() => {
            void copyToClipboard(command);
            toast.success(t("playground.commandCopied"));
          }}
        >
          <HugeiconsIcon icon={Copy01Icon} className="size-3.5" />
          {t("playground.copyCommand")}
        </Button>
      </div>
    </div>
  );
}

function SourceCard({
  source,
  onChanged,
}: {
  source: PlaygroundSource;
  onChanged: () => void;
}) {
  const t = useT();
  const [testing, setTesting] = useState(false);

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
        <span className="truncate font-medium text-sm">{source.name}</span>
        <p className="text-muted-foreground truncate text-xs">
          {source.ref}
          {source.external_model ? ` · ${source.external_model}` : ""}
        </p>
      </div>
      <div className="flex items-center gap-1.5">
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
  const [name, setName] = useState("");
  const [ref, setRef] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      await playgroundApi.createSource({
        kind: "external_openai",
        name,
        ref,
        external_model: model || null,
        api_key: apiKey || null,
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
          <p className="text-muted-foreground text-sm">{t("playground.endpointHint")}</p>
          <div className="flex flex-col gap-1.5">
            <Label>{t("playground.sourceName")}</Label>
            <Input value={name} onChange={(event) => setName(event.target.value)} />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{t("playground.baseUrl")}</Label>
            <Input
              value={ref}
              placeholder={`http://localhost:${VLLM_HINT_PORT}/v1`}
              onChange={(event) => setRef(event.target.value)}
            />
          </div>
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
