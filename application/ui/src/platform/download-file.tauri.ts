// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { downloadViaAnchor } from './download-file.shared';

// `<a download>` is ignored by WKWebView/Chromium for cross-origin URLs (it
// navigates to the URL instead of saving it). Every caller here passes a
// backend HTTP URL (`${API_BASE_URL}/...`) — model variant binaries, training
// logs, dataset export zips, dataset media items — which is a different
// origin from the Tauri webview, so the anchor flow can't be used directly.

export const downloadFile = (url: string, name?: string): void => {
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

        if (url.startsWith('blob:')) {
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        }
    } catch (error) {
        console.error('[tauri downloadFile] failed', error);
    }
};
