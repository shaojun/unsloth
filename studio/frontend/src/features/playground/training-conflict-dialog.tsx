// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState } from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useT } from "@/i18n";
import {
  usePlaygroundTrainingConflictStore,
} from "./training-conflict-store";

/** The training-start GPU-contention confirm modal. Rendered by StudioPage. */
export function PlaygroundTrainingConflictDialog() {
  const t = useT();
  const open = usePlaygroundTrainingConflictStore((state) => state.open);
  const instanceNames = usePlaygroundTrainingConflictStore(
    (state) => state.instanceNames,
  );
  const settle = usePlaygroundTrainingConflictStore((state) => state.settle);
  const [, setBusy] = useState(false);

  return (
    <AlertDialog open={open}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>
            {t("playground.trainingConflictTitle")}
          </AlertDialogTitle>
          <AlertDialogDescription>
            {t("playground.trainingConflictBody", {
              count: instanceNames.length,
              names: instanceNames.join(", "),
            })}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter className="flex-col gap-2 sm:flex-col sm:items-stretch">
          <AlertDialogAction
            className="bg-primary text-primary-foreground"
            onClick={() => {
              setBusy(true);
              settle("stop_all");
            }}
          >
            {t("playground.trainingConflictStopAll")}
          </AlertDialogAction>
          <AlertDialogAction
            onClick={() => {
              setBusy(true);
              settle("keep_running");
            }}
          >
            {t("playground.trainingConflictKeepRunning")}
          </AlertDialogAction>
          <AlertDialogCancel onClick={() => settle("cancel")}>
            {t("playground.trainingConflictCancel")}
          </AlertDialogCancel>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
