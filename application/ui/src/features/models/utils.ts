// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Model } from '../../constants/shared-types';

export const getAllModelsWithOpenVinoQuantizedModels = (models: Model[]): { id: string; name: string }[] => {
    return models.reduce<{ id: string; name: string }[]>((acc, model) => {
        const openVinoQuantizedModels = model.variants
            .filter((modelVariant) => modelVariant.format === 'openvino')
            .map((modelVariant) => ({
                id: modelVariant.id,
                name: `${model.name} [${modelVariant.precision.toUpperCase()}]`,
            }));

        acc.push(...[{ id: model.id, name: model.name }, ...openVinoQuantizedModels]);

        return acc;
    }, []);
};
