// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedDatasetRevision } from 'mocks/mock-dataset-revision';
import { getMockedModel } from 'mocks/mock-model';
import { getMockedVariant } from 'mocks/mock-model-variant';

import { sortModels } from './sorting';

describe('sortModels', () => {
    describe('name', () => {
        it('returns empty array for empty models', () => {
            const result = sortModels([], 'name', []);

            expect(result).toEqual([]);
        });

        it('sorts models by name ascending', () => {
            const models = [
                getMockedModel({ id: 'charlie', name: 'Charlie' }),
                getMockedModel({ id: 'alpha', name: 'Alpha' }),
                getMockedModel({ id: 'bravo', name: 'Bravo' }),
            ];

            const sorted = sortModels(models, 'name', []);

            expect(sorted[0].id).toBe('alpha');
            expect(sorted[1].id).toBe('bravo');
            expect(sorted[2].id).toBe('charlie');
        });

        it('sorts models by name case-insensitively', () => {
            const models = [
                getMockedModel({ id: 'upper', name: 'ZEBRA' }),
                getMockedModel({ id: 'lower', name: 'apple' }),
            ];

            const sorted = sortModels(models, 'name', []);

            expect(sorted[0].id).toBe('lower');
            expect(sorted[1].id).toBe('upper');
        });
    });

    describe('architecture', () => {
        it('sorts models by architecture ascending', () => {
            const models = [
                getMockedModel({ id: 'model-1', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-2', architecture: 'MobileNet' }),
                getMockedModel({ id: 'model-3', architecture: 'ResNet' }),
            ];

            const sorted = sortModels(models, 'architecture', []);

            expect(sorted[0].architecture).toBe('MobileNet');
            expect(sorted[1].architecture).toBe('ResNet');
            expect(sorted[2].architecture).toBe('YOLOX');
        });

        it('sorts models by architecture case-insensitively', () => {
            const models = [
                getMockedModel({ id: 'upper', architecture: 'YOLOX' }),
                getMockedModel({ id: 'lower', architecture: 'alexnet' }),
            ];

            const sorted = sortModels(models, 'architecture', []);

            expect(sorted[0].id).toBe('lower');
            expect(sorted[1].id).toBe('upper');
        });
    });

    describe('trained', () => {
        it('sorts models by training end time descending (most recent first)', () => {
            const models = [
                getMockedModel({
                    id: 'oldest',
                    training_info: { status: 'successful', end_time: '2025-01-01T00:00:00Z' },
                }),
                getMockedModel({
                    id: 'newest',
                    training_info: { status: 'successful', end_time: '2025-03-01T00:00:00Z' },
                }),
                getMockedModel({
                    id: 'middle',
                    training_info: { status: 'successful', end_time: '2025-02-01T00:00:00Z' },
                }),
            ];

            const sorted = sortModels(models, 'trained', []);

            expect(sorted[0].id).toBe('newest');
            expect(sorted[1].id).toBe('middle');
            expect(sorted[2].id).toBe('oldest');
        });

        it('places models with no end_time last (value 0)', () => {
            const models = [
                getMockedModel({ id: 'no-date', training_info: { status: 'not_started', end_time: null } }),
                getMockedModel({
                    id: 'has-date',
                    training_info: { status: 'successful', end_time: '2025-01-01T00:00:00Z' },
                }),
            ];

            const sorted = sortModels(models, 'trained', []);

            expect(sorted[0].id).toBe('has-date');
            expect(sorted[1].id).toBe('no-date');
        });

        it('places models with invalid end_time last (value 0)', () => {
            const models = [
                getMockedModel({
                    id: 'invalid-date',
                    training_info: { status: 'not_started', end_time: 'not-a-date' },
                }),
                getMockedModel({
                    id: 'valid-date',
                    training_info: { status: 'successful', end_time: '2025-06-01T00:00:00Z' },
                }),
            ];

            const sorted = sortModels(models, 'trained', []);

            expect(sorted[0].id).toBe('valid-date');
            expect(sorted[1].id).toBe('invalid-date');
        });
    });

    describe('size', () => {
        it('sorts models by size ascending', () => {
            const models = [
                getMockedModel({ id: 'large', size: 3000000 }),
                getMockedModel({ id: 'small', size: 500000 }),
                getMockedModel({ id: 'medium', size: 1500000 }),
            ];

            const sorted = sortModels(models, 'size', []);

            expect(sorted[0].id).toBe('small');
            expect(sorted[1].id).toBe('medium');
            expect(sorted[2].id).toBe('large');
        });
    });

    describe('score', () => {
        const makeModelWithScore = (id: string, primaryValue: number) =>
            getMockedModel({
                id,
                variants: [
                    getMockedVariant({
                        evaluations: [
                            {
                                dataset_revision_id: 'rev-1',
                                subset: 'testing',
                                metrics: [{ name: 'Accuracy', value: primaryValue, primary: true }],
                            },
                        ],
                    }),
                ],
            });

        it('sorts models by primary testing metric value descending', () => {
            const models = [
                makeModelWithScore('high', 0.9),
                makeModelWithScore('low', 0.5),
                makeModelWithScore('mid', 0.7),
            ];

            const sorted = sortModels(models, 'score', []);

            expect(sorted[0].id).toBe('high');
            expect(sorted[1].id).toBe('mid');
            expect(sorted[2].id).toBe('low');
        });

        it('places models with no testing metric last (value 0)', () => {
            const models = [getMockedModel({ id: 'no-metric', variants: [] }), makeModelWithScore('has-metric', 0.8)];

            const sorted = sortModels(models, 'score', []);

            expect(sorted[0].id).toBe('has-metric');
            expect(sorted[1].id).toBe('no-metric');
        });
    });

    describe('dataset', () => {
        it('sorts models by dataset revision name ascending', () => {
            const revisions = [
                getMockedDatasetRevision({ id: 'rev-a', name: 'Alpha Dataset' }),
                getMockedDatasetRevision({ id: 'rev-c', name: 'Charlie Dataset' }),
                getMockedDatasetRevision({ id: 'rev-b', name: 'Bravo Dataset' }),
            ];
            const models = [
                getMockedModel({
                    id: 'model-c',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-c' },
                }),
                getMockedModel({
                    id: 'model-a',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-a' },
                }),
                getMockedModel({
                    id: 'model-b',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-b' },
                }),
            ];

            const sorted = sortModels(models, 'dataset', revisions);

            expect(sorted[0].id).toBe('model-a');
            expect(sorted[1].id).toBe('model-b');
            expect(sorted[2].id).toBe('model-c');
        });

        it('places models with no dataset_revision_id last', () => {
            const revisions = [getMockedDatasetRevision({ id: 'rev-1', name: 'Zebra Dataset' })];
            const models = [
                getMockedModel({
                    id: 'has-dataset',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-1' },
                }),
                getMockedModel({
                    id: 'no-dataset',
                    training_info: { status: 'not_started', dataset_revision_id: null },
                }),
            ];

            const sorted = sortModels(models, 'dataset', revisions);

            expect(sorted[0].id).toBe('has-dataset');
            expect(sorted[1].id).toBe('no-dataset');
        });

        it('places models with a dataset_revision_id not found in revisions last', () => {
            const revisions = [getMockedDatasetRevision({ id: 'rev-known', name: 'Known Dataset' })];
            const models = [
                getMockedModel({
                    id: 'known',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-known' },
                }),
                getMockedModel({
                    id: 'unknown',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-unknown' },
                }),
            ];

            const sorted = sortModels(models, 'dataset', revisions);

            // 'rev-unknown' is not in the revisions map, so it sorts after 'Known Dataset'
            expect(sorted[0].id).toBe('known');
            expect(sorted[1].id).toBe('unknown');
        });

        it('returns empty array when given no models', () => {
            const revisions = [getMockedDatasetRevision()];

            expect(sortModels([], 'dataset', revisions)).toEqual([]);
        });
    });

    describe('default (unknown sort)', () => {
        it('returns the original models array and logs an error for an unknown sort option', () => {
            const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);

            const models = [getMockedModel({ id: 'a' }), getMockedModel({ id: 'b' })];
            // Cast to bypass TypeScript's exhaustive check — tests the runtime default branch
            const result = sortModels(models, 'unknown' as never, []);

            expect(result).toBe(models);
            expect(errorSpy).toHaveBeenCalledWith('Unknown sort option: unknown');

            errorSpy.mockRestore();
        });
    });
});
