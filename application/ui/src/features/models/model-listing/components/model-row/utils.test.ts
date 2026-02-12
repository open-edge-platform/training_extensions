// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedModel } from 'mocks/mock-model';
import { describe, expect, it } from 'vitest';

import { getTestingMetric } from './utils';

describe('getTestingMetric', () => {
    it('returns undefined when model has no evaluations', () => {
        const model = getMockedModel({ evaluations: [] });

        const result = getTestingMetric(model);

        expect(result).toBeUndefined();
    });

    it('returns undefined when model has no testing subset evaluation', () => {
        const model = getMockedModel({
            evaluations: [
                {
                    dataset_revision_id: '3c6c6d38-1cd8-4458-b759-b9880c048b78',
                    subset: 'training',
                    metrics: [{ name: 'Accuracy', value: 0.97, primary: true }],
                },
                {
                    dataset_revision_id: '3c6c6d38-1cd8-4458-b759-b9880c048b78',
                    subset: 'validation',
                    metrics: [{ name: 'Accuracy', value: 0.95, primary: true }],
                },
            ],
        });

        const result = getTestingMetric(model);

        expect(result).toBeUndefined();
    });

    it('returns undefined when testing subset has no primary metric', () => {
        const model = getMockedModel({
            evaluations: [
                {
                    dataset_revision_id: '3c6c6d38-1cd8-4458-b759-b9880c048b78',
                    subset: 'testing',
                    metrics: [
                        { name: 'Precision', value: 0.98, primary: false },
                        { name: 'Recall', value: 0.94, primary: false },
                    ],
                },
            ],
        });

        const result = getTestingMetric(model);

        expect(result).toBeUndefined();
    });

    it('returns testing metric when model has testing subset with primary metric', () => {
        const model = getMockedModel({
            evaluations: [
                {
                    dataset_revision_id: '3c6c6d38-1cd8-4458-b759-b9880c048b78',
                    subset: 'testing',
                    metrics: [
                        { name: 'Accuracy', value: 0.97, primary: true },
                        { name: 'Precision', value: 0.98, primary: false },
                        { name: 'Recall', value: 0.94, primary: false },
                    ],
                },
            ],
        });

        const result = getTestingMetric(model);

        expect(result).toEqual({ name: 'Accuracy', value: 97 });
    });

    it('rounds metric value to nearest integer', () => {
        const model = getMockedModel({
            evaluations: [
                {
                    dataset_revision_id: '3c6c6d38-1cd8-4458-b759-b9880c048b78',
                    subset: 'testing',
                    metrics: [{ name: 'Accuracy', value: 0.8547, primary: true }],
                },
            ],
        });

        const result = getTestingMetric(model);

        expect(result).toEqual({ name: 'Accuracy', value: 85 });
    });

    it('uses testing subset when multiple subsets exist', () => {
        const model = getMockedModel({
            evaluations: [
                {
                    dataset_revision_id: '3c6c6d38-1cd8-4458-b759-b9880c048b78',
                    subset: 'training',
                    metrics: [{ name: 'Accuracy', value: 0.99, primary: true }],
                },
                {
                    dataset_revision_id: '3c6c6d38-1cd8-4458-b759-b9880c048b78',
                    subset: 'testing',
                    metrics: [{ name: 'Accuracy', value: 0.87, primary: true }],
                },
                {
                    dataset_revision_id: '3c6c6d38-1cd8-4458-b759-b9880c048b78',
                    subset: 'validation',
                    metrics: [{ name: 'Accuracy', value: 0.95, primary: true }],
                },
            ],
        });

        const result = getTestingMetric(model);

        expect(result).toEqual({ name: 'Accuracy', value: 87 });
    });
});
