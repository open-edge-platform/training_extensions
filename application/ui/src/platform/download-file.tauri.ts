// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { save } from '@tauri-apps/plugin-dialog';
import { writeFile } from '@tauri-apps/plugin-fs';

import { toast } from '../components/toast/toast.component';

export const downloadFile = (url: string, name?: string, startedMessage?: string): void => {
    void saveDownload(url, name, startedMessage);
};

const saveDownload = async (url: string, name?: string, startedMessage?: string): Promise<void> => {
    try {
        const filename = name ?? getFallbackFilename(url);
        const selectedPath = await save({
            defaultPath: filename,
            filters: [{ name: 'All Files', extensions: ['*'] }],
        });


        if (selectedPath === null) {
            return;
        }

        if (startedMessage !== undefined) {
            toast({ type: 'info', message: startedMessage });
        }

        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`download failed: ${response.status} ${response.statusText}`);
        }

        const fileData = new Uint8Array(await response.arrayBuffer());
        await writeFile(selectedPath, fileData);
    } catch (error: unknown) {
        console.error('[tauri downloadFile] failed', error);
        toast({ type: 'error', message: 'Failed to download file' });
    }
};

const getFallbackFilename = (url: string): string => {
    if (url.startsWith('blob:')) {
        return 'download';
    }

    // Asset download URLs end with /binary, so the previous segment is usually the item id.
    const segments = new URL(url, window.location.origin).pathname.split('/').filter(Boolean);

    return segments.at(-2) ?? 'download';
};
