// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_CUSTOMIZATION,
  type SidebarNavItemPref,
  migrateShippedSidebarNavDefault,
  sanitizeCustomization,
} from "../src/features/settings/stores/appearance-custom-store.ts";

import { readSrcAsync } from "./helpers/kit.ts";

const shippedLayouts: SidebarNavItemPref[][] = [
  [
    { id: "projects", pinned: true },
    { id: "hub", pinned: true },
    { id: "images", pinned: true },
    { id: "train", pinned: true },
    { id: "video", pinned: false },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
  ],
  [
    { id: "projects", pinned: true },
    { id: "hub", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: true },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
  ],
  [
    { id: "hub", pinned: true },
    { id: "projects", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: true },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
  ],
  [
    { id: "hub", pinned: true },
  [
    { id: "hub", pinned: true },
    { id: "projects", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: false },
    { id: "audio", pinned: false },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
    { id: "battleground", pinned: true },
    { id: "api", pinned: false },
  ],
  [
    { id: "hub", pinned: true },
    { id: "projects", pinned: true },
    { id: "library", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: false },
    { id: "audio", pinned: false },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
    { id: "api", pinned: false },
  ],
    { id: "projects", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: false },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
    { id: "api", pinned: false },
  ],
  [
    { id: "hub", pinned: true },
    { id: "projects", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: false },
    { id: "audio", pinned: false },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
    { id: "api", pinned: false },
  ],
  [
    { id: "hub", pinned: true },
    { id: "projects", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: true },
    { id: "audio", pinned: false },
    { id: "train", pinned: true },
    { id: "recipes", pinned: false },
    { id: "export", pinned: false },
    { id: "api", pinned: false },
  ],
];

test("every previously shipped sidebar adopts the current default", () => {
  for (const sidebarNav of shippedLayouts) {
    const customization = sanitizeCustomization({ sidebarNav });
    assert.deepEqual(
      migrateShippedSidebarNavDefault(customization, 4, 9).sidebarNav,
      DEFAULT_CUSTOMIZATION.sidebarNav,
    );
  }
});

test("the untouched pre-Audio version-5 sidebar adopts the Audio-aware default", () => {
  const customization = sanitizeCustomization({
    sidebarNav: shippedLayouts[3],
  });
  assert.deepEqual(
    migrateShippedSidebarNavDefault(customization, 5, 9).sidebarNav,
    DEFAULT_CUSTOMIZATION.sidebarNav,
  );
});

test("a customized version-5 sidebar keeps its order and only gains the rows added since", () => {
  const customizedV5: SidebarNavItemPref[] = [
    { id: "projects", pinned: true },
    { id: "hub", pinned: true },
    { id: "train", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: false },
    { id: "recipes", pinned: true },
    { id: "export", pinned: false },
    { id: "api", pinned: false },
  ];
  const customization = sanitizeCustomization({ sidebarNav: customizedV5 });

  assert.strictEqual(
    migrateShippedSidebarNavDefault(customization, 5, 9),
    customization,
  );
  // Rows added after this customized layout are appended in current ID order.
  assert.deepEqual(
    customization.sidebarNav.map((item) => item.id),
    [...customizedV5.map((item) => item.id), "library", "audio", "battleground", "playground"],
  );
  assert.equal(
    customization.sidebarNav.find((item) => item.id === "audio")?.pinned,
    false,
  );
});

test("a user-arranged sidebar survives the migration", () => {
  const customization = sanitizeCustomization({
    sidebarNav: DEFAULT_CUSTOMIZATION.sidebarNav.map((item) =>
      item.id === "recipes" ? { ...item, pinned: true } : item,
    ),
  });
  assert.strictEqual(
    migrateShippedSidebarNavDefault(customization, 4, 9),
    customization,
  );
});

test("version-6 installs adopt the combined sidebar default", () => {
  const customization = sanitizeCustomization({
    sidebarNav: shippedLayouts[4],
  });
  const migrated = migrateShippedSidebarNavDefault(customization, 6, 9);
  assert.deepEqual(migrated.sidebarNav, DEFAULT_CUSTOMIZATION.sidebarNav);
  const ids = migrated.sidebarNav
    .filter((item) => item.pinned)
    .map((i) => i.id);
  assert.deepEqual(ids, [
    "hub",
    "projects",
    "library",
    "images",
    "train",
    "battleground",
    "playground",
  ]);
});

test("version-7 installs adopt the combined sidebar default", () => {
  const customization = sanitizeCustomization({ sidebarNav: shippedLayouts[5] });
  const migrated = migrateShippedSidebarNavDefault(customization, 7, 9);
  assert.deepEqual(migrated.sidebarNav, DEFAULT_CUSTOMIZATION.sidebarNav);
  assert.equal(migrated.sidebarNav.find((item) => item.id === "video")?.pinned, false);
});

test("both version-8 sidebar variants adopt the combined default", () => {
  for (const sidebarNav of shippedLayouts.slice(6)) {
    const customization = sanitizeCustomization({ sidebarNav });
    assert.deepEqual(
      migrateShippedSidebarNavDefault(customization, 8, 9).sidebarNav,
      DEFAULT_CUSTOMIZATION.sidebarNav,
    );
  }
});

test("a shipped-looking layout chosen after migration is preserved", () => {
  const customization = sanitizeCustomization({
    sidebarNav: shippedLayouts[2],
  });
  assert.strictEqual(
    migrateShippedSidebarNavDefault(customization, 9, 9),
    customization,
  );
});

/** The sync module uses path aliases, so read its constant rather than import it. */
async function personalizationVersion(): Promise<number> {
  const source = await readSrcAsync("features/profile/hooks/use-personalization-sync.ts");
  const match = /PERSONALIZATION_VERSION = (\d+)/.exec(source);
  assert.ok(match, "no PERSONALIZATION_VERSION in the sync module");
  return Number(match[1]);
}

test("a synced profile picks the layout change up too", async () => {
  // Remote hydration replaces the local store wholesale, so a nav default that
  // only migrates locally is overwritten by the stored layout on every login.
  // PERSONALIZATION_VERSION has to move with the layout for that migration to
  // run against the remote record.
  const stored = sanitizeCustomization({ sidebarNav: shippedLayouts[7] });
  const migrated = migrateShippedSidebarNavDefault(
    stored,
    4,
    await personalizationVersion(),
  );
  assert.deepEqual(migrated.sidebarNav, DEFAULT_CUSTOMIZATION.sidebarNav);
});
