// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from '@testing-library/react';
import { getMockedModel } from 'mocks/mock-model';

import { useGroupedModels } from './use-grouped-models.hook';

const mockActiveModelId = vi.hoisted(() => vi.fn<() => string | undefined>(() => undefined));

vi.mock('../../hooks/api/use-get-active-model-architecture-id.hook', () => ({
    useGetActiveModelArchitectureId: mockActiveModelId,
}));

describe('useGroupedModels', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe('basic functionality', () => {
        it('should return empty array when models is undefined', () => {
            const { result } = renderHook(() =>
                useGroupedModels(undefined, { groupBy: 'dataset', sortBy: 'name', pinActive: false, searchBy: '' })
            );

            expect(result.current).toEqual([]);
        });

        it('should return empty array when models is empty', () => {
            const { result } = renderHook(() =>
                useGroupedModels([], { groupBy: 'dataset', sortBy: 'name', pinActive: false, searchBy: '' })
            );

            expect(result.current).toEqual([]);
        });
    });

    describe('grouping by dataset', () => {
        it('should group models by dataset', () => {
            const models = [
                getMockedModel({
                    id: 'model-1',
                    training_info: {
                        status: 'successful',
                        dataset_revision_id: 'dataset-1',
                        label_schema_revision: {},
                        configuration: {},
                    },
                }),
                getMockedModel({
                    id: 'model-2',
                    training_info: {
                        status: 'successful',
                        dataset_revision_id: 'dataset-1',
                        label_schema_revision: {},
                        configuration: {},
                    },
                }),
                getMockedModel({
                    id: 'model-3',
                    training_info: {
                        status: 'successful',
                        dataset_revision_id: 'dataset-2',
                        label_schema_revision: {},
                        configuration: {},
                    },
                }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, { groupBy: 'dataset', sortBy: 'name', pinActive: false, searchBy: '' })
            );

            expect(result.current).toHaveLength(2);
            expect(result.current[0].models).toHaveLength(2);
            expect(result.current[1].models).toHaveLength(1);
        });
    });

    describe('grouping by architecture', () => {
        it('should group models by architecture', () => {
            const models = [
                getMockedModel({ id: 'model-1', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-2', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-3', architecture: 'MobileNet' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, { groupBy: 'architecture', sortBy: 'name', pinActive: false, searchBy: '' })
            );

            expect(result.current).toHaveLength(2);

            const yoloxGroup = result.current.find(({ group }) => group.name === 'YOLOX');
            const mobileNetGroup = result.current.find(({ group }) => group.name === 'MobileNet');

            expect(yoloxGroup?.models).toHaveLength(2);
            expect(mobileNetGroup?.models).toHaveLength(1);
        });
    });

    describe('pinning active model', () => {
        it('should not pin active model when pinActive is false', () => {
            mockActiveModelId.mockReturnValue('model-3');

            const models = [
                getMockedModel({ id: 'model-1', name: 'Model A', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-2', name: 'Model B', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-3', name: 'Model C', architecture: 'YOLOX' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, { groupBy: 'architecture', sortBy: 'name', pinActive: false, searchBy: '' })
            );

            expect(result.current[0].models[0].id).toBe('model-1');
            expect(result.current[0].models[1].id).toBe('model-2');
            expect(result.current[0].models[2].id).toBe('model-3');
        });

        it('should pin active model to first position when pinActive is true', () => {
            mockActiveModelId.mockReturnValue('model-3');

            const models = [
                getMockedModel({ id: 'model-1', name: 'Model A', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-2', name: 'Model B', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-3', name: 'Model C', architecture: 'YOLOX' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, { groupBy: 'architecture', sortBy: 'name', pinActive: true, searchBy: '' })
            );

            expect(result.current[0].models[0].id).toBe('model-3');
            expect(result.current[0].models[1].id).toBe('model-1');
            expect(result.current[0].models[2].id).toBe('model-2');
        });

        it('should maintain sorted order for non-active models when pinning active', () => {
            mockActiveModelId.mockReturnValue('bravo');

            const models = [
                getMockedModel({ id: 'charlie', name: 'Charlie Model', architecture: 'YOLOX' }),
                getMockedModel({ id: 'alpha', name: 'Alpha Model', architecture: 'YOLOX' }),
                getMockedModel({ id: 'bravo', name: 'Bravo Model', architecture: 'YOLOX' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, { groupBy: 'architecture', sortBy: 'name', pinActive: true, searchBy: '' })
            );

            expect(result.current[0].models[0].id).toBe('bravo');
            expect(result.current[0].models[1].id).toBe('alpha');
            expect(result.current[0].models[2].id).toBe('charlie');
        });
    });

    describe('search filtering', () => {
        it('should return all models when searchBy is empty', () => {
            const models = [
                getMockedModel({ id: 'model-1', name: 'ResNet-50' }),
                getMockedModel({ id: 'model-2', name: 'YOLOX-S' }),
                getMockedModel({ id: 'model-3', name: 'MobileNet-V2' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, { groupBy: 'architecture', sortBy: 'name', pinActive: false, searchBy: '' })
            );

            const allModels = result.current.flatMap((group) => group.models);
            expect(allModels).toHaveLength(3);
        });

        it('should filter models by name (case-insensitive)', () => {
            const models = [
                getMockedModel({ id: 'model-1', name: 'ResNet-50', architecture: 'ResNet' }),
                getMockedModel({ id: 'model-2', name: 'YOLOX-S', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-3', name: 'resnet-101', architecture: 'ResNet' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'architecture',
                    sortBy: 'name',
                    pinActive: false,
                    searchBy: 'resnet',
                })
            );

            const allModels = result.current.flatMap((group) => group.models);
            expect(allModels).toHaveLength(2);
            expect(allModels.map((m) => m.name)).toEqual(expect.arrayContaining(['ResNet-50', 'resnet-101']));
        });

        it('should return empty groups when no models match search query', () => {
            const models = [
                getMockedModel({ id: 'model-1', name: 'ResNet-50' }),
                getMockedModel({ id: 'model-2', name: 'YOLOX-S' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'architecture',
                    sortBy: 'name',
                    pinActive: false,
                    searchBy: 'nonexistent',
                })
            );

            expect(result.current).toHaveLength(0);
        });

        it('should filter out empty groups after search', () => {
            const models = [
                getMockedModel({ id: 'model-1', name: 'ResNet-50', architecture: 'ResNet' }),
                getMockedModel({ id: 'model-2', name: 'YOLOX-S', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-3', name: 'ResNet-101', architecture: 'ResNet' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'architecture',
                    sortBy: 'name',
                    pinActive: false,
                    searchBy: 'YOLOX',
                })
            );

            // Only YOLOX group should remain, ResNet group should be filtered out
            expect(result.current).toHaveLength(1);
            expect(result.current[0].group.name).toBe('YOLOX');
        });

        it('should apply search filter before grouping and sorting', () => {
            const models = [
                getMockedModel({ id: 'model-1', name: 'Alpha-ResNet', architecture: 'ResNet' }),
                getMockedModel({ id: 'model-2', name: 'Beta-YOLOX', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-3', name: 'Gamma-ResNet', architecture: 'ResNet' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'architecture',
                    sortBy: 'name',
                    pinActive: false,
                    searchBy: 'ResNet',
                })
            );

            expect(result.current).toHaveLength(1);
            expect(result.current[0].models).toHaveLength(2);

            expect(result.current[0].models[0].name).toBe('Alpha-ResNet');
            expect(result.current[0].models[1].name).toBe('Gamma-ResNet');
        });

        it('should work with pinActive when searching', () => {
            mockActiveModelId.mockReturnValue('model-3');

            const models = [
                getMockedModel({ id: 'model-1', name: 'ResNet-A', architecture: 'ResNet' }),
                getMockedModel({ id: 'model-2', name: 'YOLOX-B', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-3', name: 'ResNet-C', architecture: 'ResNet' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'architecture',
                    sortBy: 'name',
                    pinActive: true,
                    searchBy: 'ResNet',
                })
            );

            expect(result.current).toHaveLength(1);
            expect(result.current[0].models[0].id).toBe('model-3');
            expect(result.current[0].models[1].id).toBe('model-1');
        });

        it('should match partial names', () => {
            const models = [
                getMockedModel({ id: 'model-1', name: 'My-Custom-Model-v1' }),
                getMockedModel({ id: 'model-2', name: 'Another-Model' }),
                getMockedModel({ id: 'model-3', name: 'Custom-Detection' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'architecture',
                    sortBy: 'name',
                    pinActive: false,
                    searchBy: 'Custom',
                })
            );

            const allModels = result.current.flatMap((group) => group.models);
            expect(allModels).toHaveLength(2);
            expect(allModels.map((m) => m.name)).toEqual(
                expect.arrayContaining(['My-Custom-Model-v1', 'Custom-Detection'])
            );
        });
    });
});
