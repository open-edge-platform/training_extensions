// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const getMockedAnnotation = <T>(annotation?: Partial<T>): T => {
    return {
        id: 'annotation-1',
        shape: {
            shapeType: 'rect',
            x: 10,
            y: 20,
            width: 100,
            height: 50,
        },
        labels: [
            {
                color: '#ffff00',
                id: 'label-1',
                name: 'label-1',
                isPrediction: false,
            },
        ],
        ...annotation,
    } as T;
};
