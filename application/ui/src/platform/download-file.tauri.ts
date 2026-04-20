// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// `<a download>` is ignored by WKWebView/Chromium for cross-origin URLs (the
// webview navigates instead of downloading), and `fetch` would be blocked by
// webview CORS. `@tauri-apps/plugin-upload` does the request in Rust, so
// neither restriction applies. Blob URLs are same-origin to the webview, so
// the standard anchor flow still works for them.

import { downloadFile as downloadBlobViaAnchor } from './download-file';

export const downloadFile = (url: string, name?: string): void => {
    if (url.startsWith('blob:')) {
        downloadBlobViaAnchor(url, name);

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

    const path = await save({ defaultPath: name });
    if (path === null) {
        return;
    }

    await download(url, path);
};
