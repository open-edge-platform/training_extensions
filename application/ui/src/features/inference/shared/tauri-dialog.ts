// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Normalizes the return value of a Tauri dialog `open()` call to a single
 * path string, or `null` when the user dismissed the dialog.
 */
export const normalizeSelectedPath = (selectedPath: string | string[] | null): string | null => {
    if (typeof selectedPath === 'string') {
        return selectedPath;
    }

    if (Array.isArray(selectedPath)) {
        return selectedPath[0] ?? null;
    }

    return null;
};
