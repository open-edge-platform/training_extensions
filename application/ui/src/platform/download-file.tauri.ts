// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { downloadViaAnchor } from './download-file.shared';

// `<a download>` is ignored by WKWebView/Chromium for cross-origin URLs (it
// navigates to the URL instead of saving it). Most callers here pass a
// backend HTTP URL (`${API_BASE_URL}/...`) — model variant binaries, dataset
// export zips, dataset media items — which is a different origin from the
// Tauri webview, so the anchor flow can't be used directly. Those URLs go
// through `autoDownload`, which fetches the response and re-wraps it in a
// same-origin `blob:` URL the anchor will save. Callers that already have a
// `blob:` URL (e.g. training logs) skip the fetch and go straight
// to the anchor.

export const downloadFile = (url: string, name?: string): void => {
    if (url.startsWith('blob:')) {
        downloadViaAnchor(url, name);
        return;
    }

    void autoDownload(url, name);
};

const autoDownload = async (url: string, name?: string): Promise<void> => {
    try {
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`download failed: ${response.status} ${response.statusText}`);
        }

        // Asset URLs end with `/binary`, so the preceding path segment is the real
        // filename (e.g. `.../my-image.png/binary`); fall back to a generic name
        // if that segment is missing.
        const segments = new URL(url).pathname.split('/').filter(Boolean);
        const filename = name ?? segments.at(-2) ?? 'download';
        const blobUrl = URL.createObjectURL(await response.blob());

        downloadViaAnchor(blobUrl, filename);
    } catch (error) {
        console.error('[tauri downloadFile] failed', error);
    }
};
