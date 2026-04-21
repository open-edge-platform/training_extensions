// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Programmatically trigger the browser's "save as" flow for `url` by clicking
// a hidden `<a download>` element. The anchor is only valid for same-origin
// URLs (or `blob:` / `data:` URIs); cross-origin HTTP URLs will navigate
// instead of downloading and must be wrapped in a blob URL by the caller.
export const downloadViaAnchor = (url: string, name?: string): void => {
    const link = document.createElement('a');

    link.href = url;

    if (name !== undefined) {
        link.download = name;
    }

    link.hidden = true;
    link.click();

    if (url.startsWith('blob:')) {
        // `link.click()` only *queues* the download; the blob URL must stay
        // valid until the browser starts reading from it, so defer cleanup.
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
};
