// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type SpectrumDropZoneProps } from '@geti/ui';

import { GetElementType } from '../../../../../shared/util';

type DropItem = GetElementType<DropEvent['items']>;
export type DropEvent = Parameters<NonNullable<SpectrumDropZoneProps['onDrop']>>[0];

const VALID_DATASET_TYPES = ['zip'];

const toArray = async <T>(asyncIterator: AsyncIterable<T>) => {
    const arr = [];
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

    const filesFromDirectory = [];
    for await (const entry of entries) {
        if (entry.kind === 'directory') {
            filesFromDirectory.push(...(await flattenDropItemToFiles(entry)));
        } else {
            filesFromDirectory.push(await entry.getFile());
        }
    }

    return filesFromDirectory;
};

export const getFilesFromDropEvent = async (e: DropEvent): Promise<File[]> => {
    const files: File[] = [];
    for await (const item of e.items) {
        files.push(...(await flattenDropItemToFiles(item)));
    }

    return files;
};

export const formatToFileArray = (files: FileList | File[] | null): File[] => {
    return (files && Array.from(files)) ?? [];
};

export const isSupportedDatasetZip = (file: File): boolean => {
    const fileType = file.type ? file.type.split('/')[1] : '';

    return fileType ? VALID_DATASET_TYPES.includes(fileType) : false;
};
