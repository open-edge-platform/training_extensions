// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../api/client';
import type { DatasetItemAnnotationStatus, DatasetSubset } from '../constants/shared-types';
import { useProjectIdentifier } from './use-project-identifier.hook';

type UseGetDatasetItemsOptions = {
    limit: number;
    annotationStatus?: DatasetItemAnnotationStatus;
    subset?: DatasetSubset;
};

export const useGetDatasetItems = (options?: UseGetDatasetItemsOptions) => {
    const project_id = useProjectIdentifier();

    return $api.useQuery('get', '/api/projects/{project_id}/dataset/items', {
        params: {
            path: {
                project_id,
            },
            query: {
                annotation_status: options?.annotationStatus,
                limit: options?.limit,
                subset: options?.subset,
            },
        },
    });
};
