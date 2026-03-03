// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import dayjs from 'dayjs';
import { orderBy } from 'lodash-es';

import type { Model } from '../../../../constants/shared-types';
import { getTestingMetric } from '../components/model-row/utils';
import type { SortBy } from '../types';

export const sortModels = (models: Model[], sortBy: SortBy): Model[] => {
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
            return orderBy(models, (model) => model.size ?? 0, 'asc');
        case 'score':
            return orderBy(models, (model) => getTestingMetric(model)?.value ?? 0, 'asc');
        default:
            return models;
    }
};
