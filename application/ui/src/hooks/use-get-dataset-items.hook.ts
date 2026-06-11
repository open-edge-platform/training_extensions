// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { $api } from '../api/client';
import type { DatasetItemAnnotationStatus, DatasetSubset, Pagination } from '../constants/shared-types';
import { useProjectIdentifier } from './use-project-identifier.hook';

const DATASET_ITEMS_LIMIT = 40;

type UseGetDatasetItemsOptions = {
    subset?: DatasetSubset;
    annotationStatus?: DatasetItemAnnotationStatus;
};

export const useGetDatasetItems = ({ subset, annotationStatus }: UseGetDatasetItemsOptions = {}) => {
    const project_id = useProjectIdentifier();

    const query: {
        offset: number;
        limit: number;
        subset?: DatasetSubset;
        annotation_status?: DatasetItemAnnotationStatus;
    } = {
        offset: 0,
        limit: DATASET_ITEMS_LIMIT,
    };

    if (subset !== undefined) {
        query.subset = subset;
    }

    if (annotationStatus !== undefined) {
        query.annotation_status = annotationStatus;
    }

    const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isPending } = $api.useInfiniteQuery(
        'get',
        '/api/projects/{project_id}/dataset/items',
        {
            params: {
                query,
                path: { project_id },
            },
        },
        {
            pageParamName: 'offset',
            getNextPageParam: ({ pagination }: { pagination: Pagination }) => {
                const total = pagination.offset + pagination.count;

                if (total >= pagination.total) {
                    return undefined;
                }

                return pagination.offset + DATASET_ITEMS_LIMIT;
            },
        }
    );

    const items = useMemo(() => {
        return data?.pages.flatMap((page) => page.items) ?? [];
    }, [data?.pages]);

    const totalCount = data?.pages[0]?.pagination?.total ?? 0;

    return { items, fetchNextPage, hasNextPage, isFetchingNextPage, isPending, totalCount };
};
