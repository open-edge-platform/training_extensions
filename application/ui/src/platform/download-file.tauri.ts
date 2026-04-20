// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// `<a download>` is ignored by WKWebView/Chromium for cross-origin URLs (the
// webview navigates instead of downloading), and `fetch` would be blocked by
// webview CORS. `@tauri-apps/plugin-upload` does the request in Rust, so
// neither restriction applies.
export const downloadFile = (url: string, name?: string): void => {
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

    if (url.startsWith('blob:')) {
        // Blob URLs only exist inside the webview; Rust can't fetch them.
        const { writeFile } = await import('@tauri-apps/plugin-fs');
        const response = await fetch(url);
        const bytes = new Uint8Array(await response.arrayBuffer());

        await writeFile(path, bytes);

        URL.revokeObjectURL(url);

        return;
    }

    await download(url, path);
};
