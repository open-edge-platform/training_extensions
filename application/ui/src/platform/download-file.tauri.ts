// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// `<a download>` is ignored by WKWebView/Chromium for cross-origin URLs (it
// navigates instead of downloading). To match the web build we fetch the
// response and expose it as a same-origin blob URL the anchor can save.
// Blob URLs strip `Content-Disposition`, so we fall back to the URL's last
// path segment for the filename. When the caller passes a name (e.g. on
// Media Gallery), we instead open a native save dialog.

import { downloadViaAnchor } from './download-file-anchor';

export const downloadFile = (url: string, name?: string): void => {
    if (url.startsWith('blob:')) {
        downloadViaAnchor(url, name);

        return;
    }

    const downloadType = name === undefined ? autoDownload(url) : downloadWithDialog(url, name);

    void downloadType.catch((error: unknown) => {
        console.error('[tauri downloadFile] failed', error);
    });
};

const autoDownload = async (url: string): Promise<void> => {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`download failed: ${response.status} ${response.statusText}`);
    }

    const segments = new URL(url).pathname.split('/').filter(Boolean);

    // The urls for downloading assets end with /binary, so we take the preceding segment as the filename.
    // If that segment is missing for some reason, we fall back to a generic name.
    const filename = segments.at(-2) ?? 'download';
    const blobUrl = URL.createObjectURL(await response.blob());

    downloadViaAnchor(blobUrl, filename);
};

const downloadWithDialog = async (url: string, name: string): Promise<void> => {
    const [{ save }, { download }, { downloadDir, join }] = await Promise.all([
        import('@tauri-apps/plugin-dialog'),
        import('@tauri-apps/plugin-upload'),
        import('@tauri-apps/api/path'),
    ]);

    const defaultPath = await join(await downloadDir(), name);

    const path = await save({ defaultPath });
    if (path === null) {
        return;
    }

    await download(url, path);
};
