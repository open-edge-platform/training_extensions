// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// `<a download>` is ignored by WKWebView/Chromium for cross-origin URLs (it
// navigates instead of downloading). To match the web build we fetch the
// response, expose it as a same-origin blob URL, and let the anchor flow
// auto-save it to the user's Downloads folder. Blob URLs strip
// `Content-Disposition`, so the caller-supplied `name` (or the URL's last
// path segment) is used as the filename.

export const downloadFile = (url: string, name?: string): void => {
    if (url.startsWith('blob:')) {
        downloadViaAnchor(url, name);

        return;
    }

    void autoDownload(url, name).catch((error: unknown) => {
        console.error('[tauri downloadFile] failed', error);
    });
};

const autoDownload = async (url: string, name?: string): Promise<void> => {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`download failed: ${response.status} ${response.statusText}`);
    }

    // The urls for downloading assets end with /binary, so we take the preceding segment as the filename.
    // If that segment is missing for some reason, we fall back to a generic name.
    const segments = new URL(url).pathname.split('/').filter(Boolean);
    const filename = name ?? segments.at(-2) ?? 'download';
    const blobUrl = URL.createObjectURL(await response.blob());

    downloadViaAnchor(blobUrl, filename);
};

const downloadViaAnchor = (url: string, name?: string): void => {
    const link = document.createElement('a');

    link.href = url;
    if (name !== undefined) {
        link.download = name;
    }
    link.hidden = true;
    link.click();

    if (url.startsWith('blob:')) {
        URL.revokeObjectURL(url);
    }
};
