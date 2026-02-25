// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { AnnotationDTO } from '../../../constants/shared-types';
import { getInitialAnnotations, getInitialPredictions } from './utils';

describe('getInitialAnnotations', () => {
    const mockAnnotations: AnnotationDTO[] = [
        {
            shape: { type: 'rectangle', x: 0, y: 0, width: 100, height: 100 },
            labels: [{ id: '1' }],
        },
        {
            shape: {
                type: 'polygon',
                points: [
                    { x: 0, y: 0 },
                    { x: 100, y: 100 },
                ],
            },
            labels: [{ id: '2' }],
        },
        {
            shape: { type: 'full_image' },
            labels: [{ id: '3' }],
        },
    ];

    it('returns annotations when user has reviewed', () => {
        const result = getInitialAnnotations(true, mockAnnotations);
        expect(result).toEqual(mockAnnotations);
    });

    it('returns empty array when user has not reviewed', () => {
        const result = getInitialAnnotations(false, mockAnnotations);
        expect(result).toEqual([]);
    });
});

describe('getInitialPredictions', () => {
    const mockAnnotations: AnnotationDTO[] = [
        {
            shape: { type: 'rectangle', x: 0, y: 0, width: 100, height: 100 },
            labels: [{ id: '1' }],
        },
        {
            shape: {
                type: 'polygon',
                points: [
                    { x: 0, y: 0 },
                    { x: 100, y: 100 },
                ],
            },
            labels: [{ id: '2' }],
        },
        {
            shape: { type: 'full_image' },
            labels: [{ id: '3' }],
        },
    ];

    it('returns empty array when user has reviewed', () => {
        const result = getInitialPredictions(true, mockAnnotations);
        expect(result).toEqual([]);
    });

    it('returns annotations when user has not reviewed', () => {
        const result = getInitialPredictions(false, mockAnnotations);
        expect(result).toEqual(mockAnnotations);
    });
});
