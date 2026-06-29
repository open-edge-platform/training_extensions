// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Model } from '../../constants/shared-types';

export type SelectableModel = { modelVariantId: string; name: string; modelId: string };

export const getModelIdentifierPayload = (model: SelectableModel): { model_id: string; model_variant_id: string } => ({
    model_id: model.modelId,
    model_variant_id: model.modelVariantId,
});

export const distributeByLargestRemainder = (values: number[], total: number): number[] => {
    const sum = values.reduce((acc, value) => acc + value, 0);

    if (sum <= 0 || total <= 0) {
        return values.map(() => 0);
    }

    const exactShares = values.map((value) => (value / sum) * total);
    const flooredShares = exactShares.map((share) => Math.floor(share));
    let remainder = total - flooredShares.reduce((acc, value) => acc + value, 0);

    const indicesByRemainder = exactShares
        .map((share, index) => ({ index, fractional: share - Math.floor(share) }))
        .sort((a, b) => b.fractional - a.fractional);

    const result = [...flooredShares];
    for (const { index } of indicesByRemainder) {
        if (remainder <= 0) break;
        result[index] += 1;
        remainder -= 1;
    }

    return result;
};

export const getAllModelsWithOpenVINOVariants = (models: Model[]): SelectableModel[] => {
    return models.flatMap((model) =>
        model.variants
            .filter((variant) => variant.format === 'openvino')
            .map(
                (variant): SelectableModel => ({
                    modelVariantId: variant.id,
                    modelId: model.id,
                    name: `${model.name} [${variant.precision.toUpperCase()}]`,
                })
            )
    );
};

export const isUltralyticsModel = (identifier: string): boolean => {
    const lowerIdentifier = identifier.toLocaleLowerCase();
    return lowerIdentifier.includes('yolo26-') || lowerIdentifier.includes('yolo11-') || lowerIdentifier.includes('yolo12-');
};
