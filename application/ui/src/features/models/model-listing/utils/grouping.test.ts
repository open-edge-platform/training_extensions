// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedModel } from 'mocks/mock-model';

import { groupModelsByArchitecture, groupModelsByDataset } from './grouping';

describe('groupModelsByDataset', () => {
    it('should return empty array for empty models', () => {
        const result = groupModelsByDataset([]);

        expect(result).toEqual([]);
    });

    it('should group models by dataset_revision_id', () => {
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

        const groupedModels = groupModelsByDataset(models);

        expect(groupedModels).toHaveLength(2);
        expect(groupedModels[0].models).toHaveLength(2);
        expect(groupedModels[1].models).toHaveLength(1);
    });
});

describe('groupModelsByArchitecture', () => {
    it('should return empty array for empty models', () => {
        const groupedModels = groupModelsByArchitecture([]);

        expect(groupedModels).toEqual([]);
    });

    it('should group models by architecture', () => {
        const models = [
            getMockedModel({ id: 'model-1', architecture: 'YOLOX' }),
            getMockedModel({ id: 'model-2', architecture: 'YOLOX' }),
            getMockedModel({ id: 'model-3', architecture: 'MobileNet' }),
        ];

        const groupedModels = groupModelsByArchitecture(models);

        expect(groupedModels).toHaveLength(2);

        const yoloxGroup = groupedModels.find(({ group }) => group.name === 'YOLOX');
        const mobileNetGroup = groupedModels.find(({ group }) => group.name === 'MobileNet');

        expect(yoloxGroup?.models).toHaveLength(2);
        expect(mobileNetGroup?.models).toHaveLength(1);
    });
});
