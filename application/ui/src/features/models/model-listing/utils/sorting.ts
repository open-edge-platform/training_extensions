// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { SchemaModelView } from '../../../../api/openapi-spec';
import type { SortBy } from '../types';

export const sortModels = (models: SchemaModelView[], sortBy: SortBy): SchemaModelView[] => {
    return [...models].sort((a, b) => {
        switch (sortBy) {
            case 'name':
                return (a.id ?? '').localeCompare(b.id ?? '');
            case 'architecture':
                return a.architecture.localeCompare(b.architecture);
            // TODO: Implement the rest of sorting based on real data
            case 'trained':
            case 'size':
            case 'score':
            default:
                return 0;
        }
    });
};
