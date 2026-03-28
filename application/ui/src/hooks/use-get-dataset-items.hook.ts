// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePrefetchQuery, useQuery } from '@tanstack/react-query';

import { $api } from '../api/client';
import type { DatasetItemAnnotationStatus, DatasetSubset } from '../constants/shared-types';
import { useProjectIdentifier } from './use-project-identifier.hook';

type UseGetDatasetItemsOptions = {
    limit: number;
    annotationStatus?: DatasetItemAnnotationStatus;
    subset?: DatasetSubset;
};

const getDatasetItemsQueryOptions = (projectId: string, options?: UseGetDatasetItemsOptions) => {
    return $api.queryOptions('get', '/api/projects/{project_id}/dataset/items', {
        params: {
            path: { project_id: projectId },
            query: {
                annotation_status: options?.annotationStatus,
                limit: options?.limit,
                subset: options?.subset,
            },
        },
    });
};

export const useGetDatasetItems = (options?: UseGetDatasetItemsOptions) => {
    const projectId = useProjectIdentifier();

    return useQuery(getDatasetItemsQueryOptions(projectId, options));
};

export const usePrefetchDatasetItems = (options?: UseGetDatasetItemsOptions) => {
    const projectId = useProjectIdentifier();

    return usePrefetchQuery(getDatasetItemsQueryOptions(projectId, options));
};
