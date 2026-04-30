// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedModel } from 'mocks/mock-model';
import { getMockedVariant } from 'mocks/mock-model-variant';

import { getAllModelsWithOpenVinoVariants, getModelIdentifierPayload, SelectableModel } from './utils';

describe('getAllModelsWithOpenVinoQuantizedModels', () => {
    it('returns empty array for empty models array', () => {
        expect(getAllModelsWithOpenVinoVariants([])).toEqual([]);
    });

    it('returns only base model entry when model has no variants', () => {
        const model = getMockedModel({ id: 'model-1', name: 'My Model', variants: [] });

        expect(getAllModelsWithOpenVinoVariants([model])).toEqual([{ id: 'model-1', name: 'My Model', type: 'base' }]);
    });

    it('returns only base model entry when model has no openvino variants', () => {
        const model = getMockedModel({
            id: 'model-1',
            name: 'My Model',
            variants: [
                getMockedVariant({ id: 'v-onnx', format: 'onnx', precision: 'fp32' }),
                getMockedVariant({ id: 'v-pytorch', format: 'pytorch', precision: 'fp32' }),
            ],
        });

        expect(getAllModelsWithOpenVinoVariants([model])).toEqual([{ id: 'model-1', name: 'My Model', type: 'base' }]);
    });

    it('returns base model + one openvino variant with precision uppercased', () => {
        const model = getMockedModel({
            id: 'model-1',
            name: 'My Model',
            variants: [getMockedVariant({ id: 'v-ov', format: 'openvino', precision: 'fp16' })],
        });

        expect(getAllModelsWithOpenVinoVariants([model])).toEqual([
            { id: 'model-1', name: 'My Model', type: 'base' },
            { id: 'v-ov', name: 'My Model [FP16]', type: 'openvino', modelId: 'model-1' },
        ]);
    });

    it('returns base model + all openvino variants in order', () => {
        const model = getMockedModel({
            id: 'model-1',
            name: 'My Model',
            variants: [
                getMockedVariant({ id: 'v-fp16', format: 'openvino', precision: 'fp16' }),
                getMockedVariant({ id: 'v-fp32', format: 'openvino', precision: 'fp32' }),
                getMockedVariant({ id: 'v-int8', format: 'openvino', precision: 'int8' }),
            ],
        });

        expect(getAllModelsWithOpenVinoVariants([model])).toEqual([
            { id: 'model-1', name: 'My Model', type: 'base' },
            { id: 'v-fp16', name: 'My Model [FP16]', type: 'openvino', modelId: 'model-1' },
            { id: 'v-fp32', name: 'My Model [FP32]', type: 'openvino', modelId: 'model-1' },
            { id: 'v-int8', name: 'My Model [INT8]', type: 'openvino', modelId: 'model-1' },
        ]);
    });

    it('returns base model + only openvino variant entries when variants are mixed', () => {
        const model = getMockedModel({
            id: 'model-1',
            name: 'My Model',
            variants: [
                getMockedVariant({ id: 'v-ov', format: 'openvino', precision: 'fp16' }),
                getMockedVariant({ id: 'v-onnx', format: 'onnx', precision: 'fp32' }),
            ],
        });

        expect(getAllModelsWithOpenVinoVariants([model])).toEqual([
            { id: 'model-1', name: 'My Model', type: 'base' },
            { id: 'v-ov', name: 'My Model [FP16]', type: 'openvino', modelId: 'model-1' },
        ]);
    });

    it('processes multiple models and preserves their order', () => {
        const modelA = getMockedModel({
            id: 'model-a',
            name: 'Model A',
            variants: [getMockedVariant({ id: 'v-a-ov', format: 'openvino', precision: 'fp16' })],
        });
        const modelB = getMockedModel({
            id: 'model-b',
            name: 'Model B',
            variants: [],
        });
        const modelC = getMockedModel({
            id: 'model-c',
            name: 'Model C',
            variants: [
                getMockedVariant({ id: 'v-c-fp32', format: 'openvino', precision: 'fp32' }),
                getMockedVariant({ id: 'v-c-int8', format: 'openvino', precision: 'int8' }),
            ],
        });

        expect(getAllModelsWithOpenVinoVariants([modelA, modelB, modelC])).toEqual([
            { id: 'model-a', name: 'Model A', type: 'base' },
            { id: 'v-a-ov', name: 'Model A [FP16]', type: 'openvino', modelId: 'model-a' },
            { id: 'model-b', name: 'Model B', type: 'base' },
            { id: 'model-c', name: 'Model C', type: 'base' },
            { id: 'v-c-fp32', name: 'Model C [FP32]', type: 'openvino', modelId: 'model-c' },
            { id: 'v-c-int8', name: 'Model C [INT8]', type: 'openvino', modelId: 'model-c' },
        ]);
    });
});

describe('getModelIdentifierPayload', () => {
    it('returns only model_id for a base model', () => {
        const baseModel: SelectableModel = { id: 'model-1', name: 'My Model', type: 'base' };

        expect(getModelIdentifierPayload(baseModel)).toEqual({ model_id: 'model-1' });
    });

    it('returns model_id (parent) and model_variant_id for an openvino variant', () => {
        const openVinoModel: SelectableModel = {
            id: 'v-ov',
            name: 'My Model [FP16]',
            type: 'openvino',
            modelId: 'model-1',
        };

        expect(getModelIdentifierPayload(openVinoModel)).toEqual({ model_id: 'model-1', model_variant_id: 'v-ov' });
    });
});
