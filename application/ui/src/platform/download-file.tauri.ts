// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// `<a download>` is ignored by WKWebView/Chromium for cross-origin URLs (the
// webview navigates instead of downloading), and `fetch` would be blocked by
// webview CORS. `@tauri-apps/plugin-upload` does the request in Rust, so
// neither restriction applies. Blob URLs are same-origin to the webview, so
// the standard anchor flow still works for them.

import { downloadViaAnchor } from './download-file-anchor';

export const downloadFile = (url: string, name?: string): void => {
    if (url.startsWith('blob:')) {
        downloadViaAnchor(url, name);

        return;
    }

    void runDownload(url, name).catch((error: unknown) => {
        console.error('[tauri downloadFile] failed', error);
    });
};

const runDownload = async (url: string, name?: string): Promise<void> => {
    const [{ save }, { download }] = await Promise.all([
        import('@tauri-apps/plugin-dialog'),
        import('@tauri-apps/plugin-upload'),
    ]);

    const path = await save({ defaultPath: name === undefined ? undefined : sanitizeFilename(name) });
    if (path === null) {
        return;
    }

    await download(url, path);
};

// Strip path separators and characters that are invalid in filenames on
// Windows (and noisy on POSIX), so a user-controlled `name` can't produce
// an invalid or confusing default path in the save dialog.
const sanitizeFilename = (name: string): string => {
    const cleaned = name
        .replace(/[<>:"/\\|?*\x00-\x1F]/g, '_')
        .replace(/^\.+/, '_')
        .replace(/[. ]+$/, '')
        .trim();

    return cleaned.length > 0 ? cleaned : 'download';
};
