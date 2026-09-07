// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

export type PlaygroundConflictChoice = "stop_all" | "keep_running" | "cancel";

interface PlaygroundTrainingConflictState {
  open: boolean;
  instanceNames: string[];
  resolve: ((choice: PlaygroundConflictChoice) => void) | null;
  show: (instanceNames: string[], resolve: (choice: PlaygroundConflictChoice) => void) => void;
  settle: (choice: PlaygroundConflictChoice) => void;
}

/**
 * Promise-based confirm dialog backing store for the training-start
 * GPU-contention gate: the backend refuses /api/train/start while playground
 * vLLM instances are hosting (409 + X-Unsloth-Conflict-Kind), and the
 * training start flow calls showPlaygroundTrainingConflict() to turn that
 * refusal into an explicit "stop them / keep them / cancel" choice.
 */
export const usePlaygroundTrainingConflictStore =
  create<PlaygroundTrainingConflictState>((set) => ({
    open: false,
    instanceNames: [],
    resolve: null,
    show: (instanceNames, resolve) =>
      set({ open: true, instanceNames, resolve }),
    settle: (choice) =>
      set((state) => {
        state.resolve?.(choice);
        return { open: false, instanceNames: [], resolve: null };
      }),
  }));

export function showPlaygroundTrainingConflict(
  instanceNames: string[],
): Promise<PlaygroundConflictChoice> {
  return new Promise((resolve) => {
    usePlaygroundTrainingConflictStore.getState().show(instanceNames, resolve);
  });
}
