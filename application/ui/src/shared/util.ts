// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isEmpty } from 'lodash-es';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type GetElementType<T extends any[]> = T extends (infer U)[] ? U : never;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type IsValidArrayType<T> = T extends any[] ? GetElementType<T> : never;
export const isNonEmptyArray = <T>(value: T): value is IsValidArrayType<T> => Array.isArray(value) && !isEmpty(value);

// Camilo
export const formatDownloadUrl = (url: string) => (url.startsWith('/') ? url : `/${url}`);

export const downloadFile = (url: string, name?: string) => {
    const link = document.createElement('a');

    if (name) {
        link.download = name;
    }

    link.href = url;
    link.hidden = true;
    link.click();

    link.remove();
};
