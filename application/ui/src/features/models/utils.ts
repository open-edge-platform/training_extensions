// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Model } from '../../constants/shared-types';

export type SelectableModel =
    | { id: string; name: string; type: 'base' }
    | { id: string; name: string; type: 'openvino'; modelId: string };

/**
 * Returns the model identifier fields required by the predict endpoint.
 * For base models only `model_id` is set. For OpenVINO variants both `model_id`
 * (the parent base model) and `model_variant_id` are set, matching the
 * `MediaListPredictionRequest` schema where `model_id` is always required.
 */
export const getModelIdentifierPayload = (
    model: SelectableModel
): { model_id: string } | { model_id: string; model_variant_id: string } =>
    model.type === 'base' ? { model_id: model.id } : { model_id: model.modelId, model_variant_id: model.id };

export const getAllModelsWithOpenVinoVariants = (models: Model[]): SelectableModel[] => {
    return models.reduce<SelectableModel[]>((acc, model) => {
        const openVinoVariants = model.variants
            .filter((modelVariant) => modelVariant.format === 'openvino')
            .map(
                (modelVariant): SelectableModel => ({
                    id: modelVariant.id,
                    type: 'openvino',
                    modelId: model.id,
                    name: `${model.name} [${modelVariant.precision.toUpperCase()}]`,
                })
            );

        acc.push(...[{ id: model.id, name: model.name, type: 'base' as const }, ...openVinoVariants]);

        return acc;
    }, []);
};
