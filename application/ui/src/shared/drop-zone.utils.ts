// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type SpectrumDropZoneProps } from '@geti/ui';

import { GetElementType } from './util';

type DropItem = GetElementType<DropEvent['items']>;
type DropEvent = Parameters<NonNullable<SpectrumDropZoneProps['onDrop']>>[0];

const toArray = async <T>(asyncIterator: AsyncIterable<T>): Promise<T[]> => {
    const arr: T[] = [];

    for await (const i of asyncIterator) arr.push(i);

    return arr;
};

const flattenDropItemToFiles = async (item: DropItem): Promise<File[]> => {
    if (item.kind === 'file') {
        const file = await item.getFile();

        return [file];
    }

    if (item.kind === 'text') {
        return [];
    }

    const entries = await toArray(item.getEntries());

    const filesFromDirectory: File[] = [];
    for await (const entry of entries) {
        if (entry.kind === 'directory') {
            filesFromDirectory.push(...(await flattenDropItemToFiles(entry)));
        } else {
            filesFromDirectory.push(await entry.getFile());
        }
    }

    return filesFromDirectory;
};

/**
 * Generic utility to extract files from a drop event (supports nested folders)
 */
export const getFilesFromDropEvent = async (event: DropEvent): Promise<File[]> => {
    const files: File[] = [];
    for await (const item of event.items) {
        files.push(...(await flattenDropItemToFiles(item)));
    }

    return files;
};
