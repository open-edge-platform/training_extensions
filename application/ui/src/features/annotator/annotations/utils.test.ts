// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedAnnotation } from 'mocks/mock-annotation';
import { getMockedAnnotationLabel } from 'mocks/mock-labels';
import { describe, expect, it } from 'vitest';

import { convertPredictionToAnnotation, isPrediction } from './utils';

describe('isPrediction', () => {
    it('returns true when probability is defined', () => {
        const label = getMockedAnnotationLabel({ probability: 0.9 });

        expect(isPrediction(label)).toBe(true);
    });

    it('returns true when probability is 0', () => {
        const label = getMockedAnnotationLabel({ probability: 0 });

        expect(isPrediction(label)).toBe(true);
    });

    it('returns false when probability is undefined', () => {
        const label = getMockedAnnotationLabel();

        expect(isPrediction(label)).toBe(false);
    });
});

describe('convertPredictionToAnnotation', () => {
    it('removes probability from all prediction labels', () => {
        const annotation = getMockedAnnotation({
            labels: [
                getMockedAnnotationLabel({ probability: 0.95 }),
                getMockedAnnotationLabel({ id: 'label-2', name: 'label-2', probability: 0.7 }),
            ],
        });

        const result = convertPredictionToAnnotation(annotation);

        result.labels.forEach((label) => {
            expect(label).not.toHaveProperty('probability');
        });
    });

    it('preserves all other label properties after conversion', () => {
        const label = getMockedAnnotationLabel({ probability: 0.85 });
        const annotation = getMockedAnnotation({ labels: [label] });

        const result = convertPredictionToAnnotation(annotation);

        const { probability: _probability, ...expectedLabel } = label;
        expect(result.labels[0]).toEqual(expectedLabel);
    });

    it('preserves annotation shape and id after conversion', () => {
        const annotation = getMockedAnnotation({
            labels: [getMockedAnnotationLabel({ probability: 0.6 })],
        });

        const result = convertPredictionToAnnotation(annotation);

        expect(result.id).toBe(annotation.id);
        expect(result.shape).toEqual(annotation.shape);
    });
});
