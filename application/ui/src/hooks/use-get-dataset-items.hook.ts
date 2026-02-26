// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../api/client';
import type { DatasetItemAnnotationStatus } from '../constants/shared-types';
import { useProjectIdentifier } from './use-project-identifier.hook';

type UseGetDatasetItemsOptions = {
    limit?: number;
    offset?: number;
    annotationStatus?: DatasetItemAnnotationStatus;
};

export const useGetDatasetItems = (options?: UseGetDatasetItemsOptions) => {
    const project_id = useProjectIdentifier();

    const query: {
        limit?: number;
        offset?: number;
        annotation_status?: DatasetItemAnnotationStatus;
    } = {};

    if (options?.limit !== undefined) {
        query.limit = options.limit;
    }

    if (options?.offset !== undefined) {
        query.offset = options.offset;
    }

    if (options?.annotationStatus !== undefined) {
        query.annotation_status = options.annotationStatus;
    }

    return $api.useQuery('get', '/api/projects/{project_id}/dataset/items', {
        params: {
            path: {
                project_id,
            },
            ...query,
        },
    });
};
