// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const getMockedLabel = <T>(label?: Partial<T>): T => {
    return {
        color: '#ffff00',
        id: 'label-1',
        name: 'label-1',
        isPrediction: false,
        ...label,
    } as T;
};
