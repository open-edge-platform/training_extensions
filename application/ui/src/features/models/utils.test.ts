// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedModel } from 'mocks/mock-model';
import { getMockedVariant } from 'mocks/mock-model-variant';

import { getAllModelsWithOpenVinoQuantizedModels } from './utils';

describe('getAllModelsWithOpenVinoQuantizedModels', () => {
    it('returns empty array for empty models array', () => {
        expect(getAllModelsWithOpenVinoQuantizedModels([])).toEqual([]);
    });

    it('returns only base model entry when model has no variants', () => {
        const model = getMockedModel({ id: 'model-1', name: 'My Model', variants: [] });

        expect(getAllModelsWithOpenVinoQuantizedModels([model])).toEqual([{ id: 'model-1', name: 'My Model' }]);
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

        expect(getAllModelsWithOpenVinoQuantizedModels([model])).toEqual([{ id: 'model-1', name: 'My Model' }]);
    });

    it('returns base model + one openvino variant with precision uppercased', () => {
        const model = getMockedModel({
            id: 'model-1',
            name: 'My Model',
            variants: [getMockedVariant({ id: 'v-ov', format: 'openvino', precision: 'fp16' })],
        });

        expect(getAllModelsWithOpenVinoQuantizedModels([model])).toEqual([
            { id: 'model-1', name: 'My Model' },
            { id: 'v-ov', name: 'My Model [FP16]' },
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

        expect(getAllModelsWithOpenVinoQuantizedModels([model])).toEqual([
            { id: 'model-1', name: 'My Model' },
            { id: 'v-fp16', name: 'My Model [FP16]' },
            { id: 'v-fp32', name: 'My Model [FP32]' },
            { id: 'v-int8', name: 'My Model [INT8]' },
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

        expect(getAllModelsWithOpenVinoQuantizedModels([model])).toEqual([
            { id: 'model-1', name: 'My Model' },
            { id: 'v-ov', name: 'My Model [FP16]' },
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

        expect(getAllModelsWithOpenVinoQuantizedModels([modelA, modelB, modelC])).toEqual([
            { id: 'model-a', name: 'Model A' },
            { id: 'v-a-ov', name: 'Model A [FP16]' },
            { id: 'model-b', name: 'Model B' },
            { id: 'model-c', name: 'Model C' },
            { id: 'v-c-fp32', name: 'Model C [FP32]' },
            { id: 'v-c-int8', name: 'Model C [INT8]' },
        ]);
    });
});
