// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  BalanceScaleIcon,
  Medal01Icon,
  ServerStack01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { AutoEvalPanel } from "./auto-eval-panel";
import { ModelsPanel } from "./models-panel";
import { TestsPanel } from "./tests-panel";

export type BattlegroundTab = "sources" | "ab" | "auto";

export function BattlegroundPage() {
  const t = useT();
  const [tab, setTab] = useState<BattlegroundTab>("sources");

  const tabs: {
    id: BattlegroundTab;
    label: string;
    icon: typeof Medal01Icon;
  }[] = [
    {
      id: "sources",
      label: t("battleground.tabSources"),
      icon: ServerStack01Icon,
    },
    { id: "ab", label: t("battleground.tabAbTests"), icon: BalanceScaleIcon },
    { id: "auto", label: t("battleground.tabAutoEval"), icon: Medal01Icon },
  ];

  return (
    <div className="flex h-full flex-1 flex-col overflow-hidden">
      <header className="flex flex-col gap-3 border-b px-4 pt-4 pb-3 sm:px-6">
        <div className="flex items-center gap-2">
          <h1 className="font-heading font-semibold text-xl tracking-tight">
            {t("battleground.title")}
          </h1>
          <span className="text-muted-foreground text-sm">
            {t("battleground.subtitle")}
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
        {tab === "sources" && <ModelsPanel />}
        {tab === "ab" && <TestsPanel />}
        {tab === "auto" && <AutoEvalPanel />}
      </div>
    </div>
  );
}
