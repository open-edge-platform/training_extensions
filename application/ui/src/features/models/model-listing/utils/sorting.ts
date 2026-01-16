// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import dayjs from 'dayjs';
import { orderBy } from 'lodash-es';

import type { SchemaModelView } from '../../../../api/openapi-spec';
import type { SortBy } from '../types';

export const sortModels = (models: SchemaModelView[], sortBy: SortBy): SchemaModelView[] => {
    switch (sortBy) {
        case 'name':
            return orderBy(models, (model) => model.name?.toLowerCase() ?? '', 'asc');
        case 'architecture':
            return orderBy(models, (model) => model.architecture?.toLowerCase() ?? '', 'asc');
        case 'trained':
            return orderBy(
                models,
                (model) => {
                    const date = dayjs(model.training_info?.end_time);

                    return date.isValid() ? date.valueOf() : 0;
                },
                'desc'
            );
        case 'size':
        // TODO: uncomment once backend returns size
        // return orderBy(models, (model) => model.size ?? 0, 'asc');
        case 'score':
        // TODO: uncomment once backend returns score
        // return orderBy(models, (model) => model.score ?? 0, 'desc');
        default:
            return models;
    }
};
