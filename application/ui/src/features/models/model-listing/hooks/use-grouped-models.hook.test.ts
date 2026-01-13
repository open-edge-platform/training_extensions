// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from '@testing-library/react';
import { getMockedModel } from 'mocks/mock-model';

import { useGroupedModels } from './use-grouped-models.hook';

const mockActiveModelId = vi.hoisted(() => vi.fn<() => string | undefined>(() => undefined));

vi.mock('../../hooks/api/use-get-active-model-id.hook', () => ({
    useGetActiveModelId: mockActiveModelId,
}));

describe('useGroupedModels', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe('basic functionality', () => {
        it('should return empty array when models is undefined', () => {
            const { result } = renderHook(() =>
                useGroupedModels(undefined, { groupBy: 'dataset', sortBy: 'name', pinActive: false })
            );

            expect(result.current).toEqual([]);
        });

        it('should return empty array when models is empty', () => {
            const { result } = renderHook(() =>
                useGroupedModels([], { groupBy: 'dataset', sortBy: 'name', pinActive: false })
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
                useGroupedModels(models, { groupBy: 'dataset', sortBy: 'name', pinActive: false })
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
                useGroupedModels(models, { groupBy: 'architecture', sortBy: 'name', pinActive: false })
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
                getMockedModel({ id: 'model-1', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-2', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-3', architecture: 'YOLOX' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, { groupBy: 'architecture', sortBy: 'name', pinActive: false })
            );

            expect(result.current[0].models[0].id).toBe('model-1');
            expect(result.current[0].models[1].id).toBe('model-2');
            expect(result.current[0].models[2].id).toBe('model-3');
        });

        it('should pin active model to first position when pinActive is true', () => {
            mockActiveModelId.mockReturnValue('model-3');

            const models = [
                getMockedModel({ id: 'model-1', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-2', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-3', architecture: 'YOLOX' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, { groupBy: 'architecture', sortBy: 'name', pinActive: true })
            );

            expect(result.current[0].models[0].id).toBe('model-3');
            expect(result.current[0].models[1].id).toBe('model-1');
            expect(result.current[0].models[2].id).toBe('model-2');
        });

        it('should maintain sorted order for non-active models when pinning active', () => {
            mockActiveModelId.mockReturnValue('bravo');

            const models = [
                getMockedModel({ id: 'charlie', architecture: 'YOLOX' }),
                getMockedModel({ id: 'alpha', architecture: 'YOLOX' }),
                getMockedModel({ id: 'bravo', architecture: 'YOLOX' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, { groupBy: 'architecture', sortBy: 'name', pinActive: true })
            );

            expect(result.current[0].models[0].id).toBe('bravo');
            expect(result.current[0].models[1].id).toBe('alpha');
            expect(result.current[0].models[2].id).toBe('charlie');
        });
    });
});
