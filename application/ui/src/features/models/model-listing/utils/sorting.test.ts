// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedModel } from 'mocks/mock-model';

import { sortModels } from './sorting';

describe('sortModels', () => {
    it('should return empty array for empty models', () => {
        const result = sortModels([], 'name');

        expect(result).toEqual([]);
    });

    it('should sort models by name ascending', () => {
        const models = [
            getMockedModel({ id: 'charlie', name: 'Charlie' }),
            getMockedModel({ id: 'alpha', name: 'Alpha' }),
            getMockedModel({ id: 'bravo', name: 'Bravo' }),
        ];

        const sorted = sortModels(models, 'name');

        expect(sorted[0].id).toBe('alpha');
        expect(sorted[1].id).toBe('bravo');
        expect(sorted[2].id).toBe('charlie');
    });

    it('should sort models by architecture ascending', () => {
        const models = [
            getMockedModel({ id: 'model-1', name: 'Charlie', architecture: 'YOLOX' }),
            getMockedModel({ id: 'model-2', name: 'Alpha', architecture: 'MobileNet' }),
            getMockedModel({ id: 'model-3', name: 'Bravo', architecture: 'ResNet' }),
        ];

        const sorted = sortModels(models, 'architecture');

        expect(sorted[0].architecture).toBe('MobileNet');
        expect(sorted[1].architecture).toBe('ResNet');
        expect(sorted[2].architecture).toBe('YOLOX');
    });
});
