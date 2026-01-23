// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isObject, isString } from 'lodash-es';

export const isErrorWithDetail = (error: unknown): error is { detail: string } => {
    return isObject(error) && 'detail' in error && isString(error.detail);
};
