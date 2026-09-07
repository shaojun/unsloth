// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AiChat02Icon,
  Chart02Icon,
  Comment01Icon,
  Medal01Icon,
  ServerStack01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { ModelsPanel } from "./models-panel";
import { PlayPanel } from "./play-panel";
import { EvaluatePanel } from "./evaluate-panel";
import { TestsPanel } from "./tests-panel";
import { ReportsPanel } from "./reports-panel";

export type PlaygroundTab = "models" | "play" | "evaluate" | "tests" | "reports";

export function PlaygroundPage() {
  const t = useT();
  const [tab, setTab] = useState<PlaygroundTab>("models");

  const tabs: { id: PlaygroundTab; label: string; icon: typeof Medal01Icon }[] = [
    { id: "models", label: t("playground.tabModels"), icon: ServerStack01Icon },
    { id: "play", label: t("playground.tabPlay"), icon: AiChat02Icon },
    { id: "evaluate", label: t("playground.tabEvaluate"), icon: Medal01Icon },
    { id: "tests", label: t("playground.tabTests"), icon: Comment01Icon },
    { id: "reports", label: t("playground.tabReports"), icon: Chart02Icon },
  ];

  return (
    <div className="flex h-full flex-1 flex-col overflow-hidden">
      <header className="flex flex-col gap-3 border-b px-4 pt-4 pb-3 sm:px-6">
        <div className="flex items-center gap-2">
          <h1 className="font-heading font-semibold text-xl tracking-tight">
            {t("playground.title")}
          </h1>
          <span className="text-muted-foreground text-sm">
            {t("playground.subtitle")}
          </span>
        </div>
        <nav className="flex flex-wrap items-center gap-1" role="tablist">
          {tabs.map(({ id, label, icon }) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={tab === id}
              onClick={() => setTab(id)}
              className={cn(
                "rounded-full px-3.5 py-1.5 text-sm font-medium transition-colors",
                tab === id
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground",
              )}
            >
              <HugeiconsIcon icon={icon} className="mr-1.5 size-3.5" />
              {label}
            </button>
          ))}
        </nav>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {tab === "models" && <ModelsPanel />}
        {tab === "play" && <PlayPanel />}
        {tab === "evaluate" && <EvaluatePanel />}
        {tab === "tests" && <TestsPanel />}
        {tab === "reports" && <ReportsPanel />}
      </div>
    </div>
  );
}
