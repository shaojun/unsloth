// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import {
  Add01Icon,
  Copy01Icon,
  Delete02Icon,
  PencilEdit02Icon,
  PlayIcon,
  RefreshIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import {
  type BattlegroundSource,
  type BattlegroundTrainedModel,
  battlegroundApi,
} from "./api/battleground-api";
import { PlayChatDialog } from "./play-chat-dialog";

/** Default port the hosting hints suggest for a manually started vLLM server. */
const VLLM_HINT_PORT = 8000;

export function ModelsPanel() {
  const t = useT();
  const [sources, setSources] = useState<BattlegroundSource[]>([]);
  const [trainedModels, setTrainedModels] = useState<
    BattlegroundTrainedModel[]
  >([]);
  const [addOpen, setAddOpen] = useState(false);
  const [playingSource, setPlayingSource] = useState<BattlegroundSource | null>(
    null,
  );

  const refresh = useCallback(async () => {
    try {
      const [src, trained] = await Promise.all([
        battlegroundApi.listSources(),
        battlegroundApi.trainedModels(),
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
          <h2 className="font-heading font-semibold text-lg">
            {t("battleground.sources")}
          </h2>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => void refresh()}>
              <HugeiconsIcon icon={RefreshIcon} className="size-4" />
            </Button>
            <Button size="sm" onClick={() => setAddOpen(true)}>
              <HugeiconsIcon icon={Add01Icon} className="size-4" />
              {t("battleground.addSource")}
            </Button>
          </div>
        </div>
        {sources.length === 0 && (
          <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
            {t("battleground.noSources")}
          </p>
        )}
        <div className="flex flex-col gap-2">
          {sources.map((source) => (
            <SourceCard
              key={source.id}
              source={source}
              onChanged={refresh}
              onPlay={() => setPlayingSource(source)}
            />
          ))}
        </div>
      </section>

      <AddSourceDialog
        open={addOpen}
        onOpenChange={setAddOpen}
        onCreated={refresh}
      />
      <PlayChatDialog
        source={playingSource}
        open={playingSource != null}
        onOpenChange={(open) => {
          if (!open) setPlayingSource(null);
        }}
      />
    </div>
  );
}

/**
 * The app never starts inference servers itself. This section tells users how
 * to host a trained model manually: copy a `vllm serve <path>` command (the
 * model's full output path is the key info) into a terminal on this machine,
 * then register the resulting endpoint as a source below.
 */
function HostingHintSection({
  models,
}: { models: BattlegroundTrainedModel[] }) {
  const t = useT();

  return (
    <section className="flex flex-col gap-3">
      <h2 className="font-heading font-semibold text-lg">
        {t("battleground.hostManuallyTitle")}
      </h2>
      <p className="text-muted-foreground text-sm">
        {t("battleground.hostManuallyIntro")}
      </p>
      <ol className="text-muted-foreground flex list-decimal flex-col gap-1 pl-5 text-sm">
        <li>{t("battleground.hostManuallyStep1")}</li>
        <li>{t("battleground.hostManuallyStep2")}</li>
        <li>{t("battleground.hostManuallyStep3")}</li>
      </ol>

      {models.length === 0 ? (
        <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
          {t("battleground.noTrainedModels")}
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

function TrainedModelRow({ model }: { model: BattlegroundTrainedModel }) {
  const t = useT();
  const command = `vllm serve "${model.output_dir}" --port ${VLLM_HINT_PORT}`;

  return (
    <div className="flex flex-col gap-2 rounded-lg border p-3">
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-sm font-medium">
          {model.name}
        </span>
        {model.model_name && model.model_name !== model.name && (
          <span className="text-muted-foreground truncate text-xs">
            {model.model_name}
          </span>
        )}
      </div>
      {model.artifact === "adapter" && (
        <p className="text-xs text-amber-500">
          {t("battleground.adapterNote")}
        </p>
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
            toast.success(t("battleground.commandCopied"));
          }}
        >
          <HugeiconsIcon icon={Copy01Icon} className="size-3.5" />
          {t("battleground.copyCommand")}
        </Button>
      </div>
    </div>
  );
}

function SourceCard({
  source,
  onChanged,
  onPlay,
}: {
  source: BattlegroundSource;
  onChanged: () => void;
  onPlay: () => void;
}) {
  const t = useT();
  const [testing, setTesting] = useState(false);
  const [editOpen, setEditOpen] = useState(false);

  const handleTest = async () => {
    setTesting(true);
    try {
      const result = await battlegroundApi.testSource(source.id);
      if (result.ok) {
        toast.success(
          t("battleground.connectionOk", { count: result.models.length }),
        );
      } else {
        toast.error(result.error || t("battleground.connectionFailed"));
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setTesting(false);
    }
  };

  const handleDelete = async () => {
    try {
      await battlegroundApi.deleteSource(source.id);
      toast.success(t("battleground.sourceDeleted"));
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
        <Button variant="outline" size="sm" onClick={onPlay}>
          <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
          {t("battleground.play")}
        </Button>
        <Button variant="outline" size="sm" onClick={() => setEditOpen(true)}>
          <HugeiconsIcon icon={PencilEdit02Icon} className="size-3.5" />
          {t("battleground.edit")}
        </Button>
        <Button
          variant="outline"
          size="sm"
          disabled={testing}
          onClick={() => void handleTest()}
        >
          {t("battleground.test")}
        </Button>
        <Button variant="ghost" size="sm" onClick={() => void handleDelete()}>
          <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
        </Button>
      </div>
      <EditSourceDialog
        source={source}
        open={editOpen}
        onOpenChange={setEditOpen}
        onSaved={onChanged}
      />
    </div>
  );
}

/** Shared form for creating and editing a source. */
function SourceFields({
  name,
  setName,
  refValue,
  setRef,
  model,
  setModel,
  apiKey,
  setApiKey,
}: {
  name: string;
  setName: (value: string) => void;
  refValue: string;
  setRef: (value: string) => void;
  model: string;
  setModel: (value: string) => void;
  apiKey: string;
  setApiKey: (value: string) => void;
}) {
  const t = useT();
  return (
    <div className="flex flex-col gap-4">
      <p className="text-muted-foreground text-sm">
        {t("battleground.endpointHint")}
      </p>
      <div className="flex flex-col gap-1.5">
        <Label>{t("battleground.sourceName")}</Label>
        <Input value={name} onChange={(event) => setName(event.target.value)} />
      </div>
      <div className="flex flex-col gap-1.5">
        <Label>{t("battleground.baseUrl")}</Label>
        <Input
          value={refValue}
          placeholder={`http://localhost:${VLLM_HINT_PORT}/v1`}
          onChange={(event) => setRef(event.target.value)}
        />
      </div>
      <div className="flex flex-col gap-1.5">
        <Label>{t("battleground.modelId")}</Label>
        <Input
          value={model}
          onChange={(event) => setModel(event.target.value)}
        />
      </div>
      <div className="flex flex-col gap-1.5">
        <Label>{t("battleground.apiKeyOptional")}</Label>
        <Input
          type="password"
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value)}
        />
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
      await battlegroundApi.createSource({
        kind: "external_openai",
        name,
        ref,
        external_model: model || null,
        api_key: apiKey || null,
      });
      toast.success(t("battleground.sourceAdded"));
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
          <DialogTitle>{t("battleground.addSource")}</DialogTitle>
        </DialogHeader>
        <SourceFields
          name={name}
          setName={setName}
          refValue={ref}
          setRef={setRef}
          model={model}
          setModel={setModel}
          apiKey={apiKey}
          setApiKey={setApiKey}
        />
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("battleground.cancel")}
          </Button>
          <Button
            disabled={!name.trim() || !ref.trim() || saving}
            onClick={() => void handleSave()}
          >
            {t("battleground.save")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function EditSourceDialog({
  source,
  open,
  onOpenChange,
  onSaved,
}: {
  source: BattlegroundSource;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSaved: () => void;
}) {
  const t = useT();
  const [name, setName] = useState(source.name);
  const [ref, setRef] = useState(source.ref);
  const [model, setModel] = useState(source.external_model ?? "");
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      await battlegroundApi.updateSource(source.id, {
        name,
        ref,
        external_model: model || null,
        ...(apiKey ? { api_key: apiKey } : {}),
      });
      toast.success(t("battleground.sourceUpdated"));
      onOpenChange(false);
      onSaved();
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
          <DialogTitle>{t("battleground.editSource")}</DialogTitle>
        </DialogHeader>
        <SourceFields
          name={name}
          setName={setName}
          refValue={ref}
          setRef={setRef}
          model={model}
          setModel={setModel}
          apiKey={apiKey}
          setApiKey={setApiKey}
        />
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("battleground.cancel")}
          </Button>
          <Button
            disabled={!name.trim() || !ref.trim() || saving}
            onClick={() => void handleSave()}
          >
            {t("battleground.save")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
