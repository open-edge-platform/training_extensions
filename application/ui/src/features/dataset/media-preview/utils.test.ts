// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { AnnotationDTO } from '../../../constants/shared-types';
import { getAnnotations } from './utils';

describe('getAnnotations', () => {
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

    describe('annotation mode', () => {
        it('returns annotations when mode is annotation and user has reviewed', () => {
            const result = getAnnotations('annotation', true, mockAnnotations);
            expect(result).toEqual(mockAnnotations);
        });

        it('returns empty array when mode is annotation and user has not reviewed', () => {
            const result = getAnnotations('annotation', false, mockAnnotations);
            expect(result).toEqual([]);
        });
    });

    describe('prediction mode', () => {
        it('returns empty array when mode is prediction and user has reviewed', () => {
            const result = getAnnotations('prediction', true, mockAnnotations);
            expect(result).toEqual([]);
        });

        it('returns annotations when mode is prediction and user has not reviewed', () => {
            const result = getAnnotations('prediction', false, mockAnnotations);
            expect(result).toEqual(mockAnnotations);
        });
    });

    describe('edge cases', () => {
        it('handles empty annotations array in annotation mode with review', () => {
            const result = getAnnotations('annotation', true, []);
            expect(result).toEqual([]);
        });

        it('handles empty annotations array in prediction mode without review', () => {
            const result = getAnnotations('prediction', false, []);
            expect(result).toEqual([]);
        });
    });
});
