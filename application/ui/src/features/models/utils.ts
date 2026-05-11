// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Model } from '../../constants/shared-types';

export type SelectableModel = { modelVariantId: string; name: string; modelId: string };

export const getModelIdentifierPayload = (model: SelectableModel): { model_id: string; model_variant_id: string } => ({
    model_id: model.modelId,
    model_variant_id: model.modelVariantId,
});

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
