// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedModel } from 'mocks/mock-model';
import { describe, expect, it } from 'vitest';

import type { Metric } from '../../../../../constants/shared-types';
import { getFirstAvailableTestingMetric, getModelEvaluations, getTestingMetric, getTestingMetrics } from './utils';

const DATASET_REVISION_ID = '3c6c6d38-1cd8-4458-b759-b9880c048b78';
const BASE_VARIANT = {
    format: 'pytorch',
    precision: 'fp32',
    weights_size: 1024,
    files_deleted: false,
} as const;

describe('getTestingMetric', () => {
    it('returns undefined when model has no evaluations', () => {
        const model = getMockedModel({ variants: [] });

        const result = getTestingMetric(model);

        expect(result).toBeUndefined();
    });

    it('returns undefined when model has no testing subset evaluation', () => {
        const model = getMockedModel({
            variants: [
                {
                    ...BASE_VARIANT,
                    id: 'variant-1',
                    evaluations: [
                        {
                            dataset_revision_id: DATASET_REVISION_ID,
                            subset: 'training',
                            metrics: [{ name: 'Accuracy', value: 0.97, primary: true }],
                        },
                        {
                            dataset_revision_id: DATASET_REVISION_ID,
                            subset: 'validation',
                            metrics: [{ name: 'Accuracy', value: 0.95, primary: true }],
                        },
                    ],
                },
            ],
        });

        const result = getTestingMetric(model);

        expect(result).toBeUndefined();
    });

    it('returns undefined when testing subset has no primary metric', () => {
        const model = getMockedModel({
            variants: [
                {
                    ...BASE_VARIANT,
                    id: 'variant-1',
                    evaluations: [
                        {
                            dataset_revision_id: DATASET_REVISION_ID,
                            subset: 'testing',
                            metrics: [
                                { name: 'Precision', value: 0.98, primary: false },
                                { name: 'Recall', value: 0.94, primary: false },
                            ],
                        },
                    ],
                },
            ],
        });

        const result = getTestingMetric(model);

        expect(result).toBeUndefined();
    });

    it('returns testing metric when model has testing subset with primary metric', () => {
        const model = getMockedModel({
            variants: [
                {
                    ...BASE_VARIANT,
                    id: 'variant-1',
                    evaluations: [
                        {
                            dataset_revision_id: DATASET_REVISION_ID,
                            subset: 'testing',
                            metrics: [
                                { name: 'Accuracy', value: 0.97, primary: true },
                                { name: 'Precision', value: 0.98, primary: false },
                                { name: 'Recall', value: 0.94, primary: false },
                            ],
                        },
                    ],
                },
            ],
        });

        const result = getTestingMetric(model);

        expect(result).toEqual({ name: 'Accuracy', value: 97 });
    });

    it('rounds metric value to nearest integer', () => {
        const model = getMockedModel({
            variants: [
                {
                    ...BASE_VARIANT,
                    id: 'variant-1',
                    evaluations: [
                        {
                            dataset_revision_id: DATASET_REVISION_ID,
                            subset: 'testing',
                            metrics: [{ name: 'Accuracy', value: 0.8547, primary: true }],
                        },
                    ],
                },
            ],
        });

        const result = getTestingMetric(model);

        expect(result).toEqual({ name: 'Accuracy', value: 85 });
    });

    it('uses testing subset when multiple subsets exist', () => {
        const model = getMockedModel({
            variants: [
                {
                    ...BASE_VARIANT,
                    id: 'variant-1',
                    evaluations: [
                        {
                            dataset_revision_id: DATASET_REVISION_ID,
                            subset: 'training',
                            metrics: [{ name: 'Accuracy', value: 0.99, primary: true }],
                        },
                        {
                            dataset_revision_id: DATASET_REVISION_ID,
                            subset: 'testing',
                            metrics: [{ name: 'Accuracy', value: 0.87, primary: true }],
                        },
                        {
                            dataset_revision_id: DATASET_REVISION_ID,
                            subset: 'validation',
                            metrics: [{ name: 'Accuracy', value: 0.95, primary: true }],
                        },
                    ],
                },
            ],
        });

        const result = getTestingMetric(model);

        expect(result).toEqual({ name: 'Accuracy', value: 87 });
    });
});

describe('getTestingMetrics', () => {
    it('returns empty array when there is no testing evaluation', () => {
        const model = getMockedModel({
            variants: [
                {
                    ...BASE_VARIANT,
                    id: 'variant-1',
                    evaluations: [
                        {
                            dataset_revision_id: DATASET_REVISION_ID,
                            subset: 'training',
                            metrics: [{ name: 'Accuracy', value: 0.12, primary: true }],
                        },
                    ],
                },
            ],
        });

        expect(getTestingMetrics(getModelEvaluations(model))).toEqual([]);
    });

    it('returns all metrics from testing evaluation', () => {
        const testingMetrics: Metric[] = [
            { name: 'Accuracy', value: 0.91, primary: true },
            { name: 'Precision', value: 0.89, primary: false },
            { name: 'Recall', value: 0.87, primary: false },
        ];

        const model = getMockedModel({
            variants: [
                {
                    ...BASE_VARIANT,
                    id: 'variant-1',
                    evaluations: [
                        {
                            dataset_revision_id: DATASET_REVISION_ID,
                            subset: 'testing',
                            metrics: testingMetrics,
                        },
                    ],
                },
            ],
        });

        expect(getTestingMetrics(getModelEvaluations(model))).toEqual(testingMetrics);
    });
});

describe('getFirstAvailableTestingMetric', () => {
    it('returns undefined for undefined models', () => {
        expect(getFirstAvailableTestingMetric(undefined)).toBeUndefined();
    });

    it('returns first model primary testing metric', () => {
        const modelWithoutPrimary = getMockedModel({
            variants: [
                {
                    ...BASE_VARIANT,
                    id: 'variant-1',
                    evaluations: [
                        {
                            dataset_revision_id: DATASET_REVISION_ID,
                            subset: 'testing',
                            metrics: [{ name: 'Accuracy', value: 0.9, primary: false }],
                        },
                    ],
                },
            ],
        });

        const modelWithPrimary = getMockedModel({
            variants: [
                {
                    ...BASE_VARIANT,
                    id: 'variant-2',
                    evaluations: [
                        {
                            dataset_revision_id: DATASET_REVISION_ID,
                            subset: 'testing',
                            metrics: [{ name: 'mAP', value: 0.923, primary: true }],
                        },
                    ],
                },
            ],
        });

        expect(getFirstAvailableTestingMetric([modelWithoutPrimary, modelWithPrimary])).toEqual({
            name: 'mAP',
            value: 92,
        });
    });
});
