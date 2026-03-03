// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import isObject from 'lodash-es/isObject';

export const isInvalidStagedFile = (error: unknown): boolean => {
    if (isObject(error) && 'detail' in error) {
        const detail = String(error.detail);
        return detail.includes('not found');
    }

    return false;
};
