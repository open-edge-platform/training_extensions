// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Default (web) implementation: nothing to clean up on close because
// browsers tear down the tab on their own. The Tauri build replaces this
// file with `storage-cleanup.tauri.ts` via `resolve.extensions`.
export const setupStorageCleanup = (): void => {};
