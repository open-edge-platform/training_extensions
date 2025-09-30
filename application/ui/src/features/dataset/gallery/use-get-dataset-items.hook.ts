// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../api/client';

const datasetItemsLimit = 20;

export const useGetDatasetItems = () => {
    const project_id = useProjectIdentifier();

    const { data, fetchNextPage, hasNextPage, isFetchingNextPage } = $api.useInfiniteQuery(
        'get',
        '/api/projects/{project_id}/dataset/items',
        {
            params: {
                query: { offset: 0, limit: datasetItemsLimit },
                path: { project_id },
            },
        },
        {
            pageParamName: 'offset',
            getNextPageParam: ({
                pagination,
            }: {
                pagination: { offset: number; limit: number; count: number; total: number };
            }) => {
                const total = pagination.offset + pagination.count;

                if (total >= pagination.total) {
                    return undefined;
                }

                return pagination.offset + datasetItemsLimit;
            },
        }
    );

    const items = data?.pages.flatMap((page) => page.items) ?? [];

    return { items, fetchNextPage, hasNextPage, isFetchingNextPage };
};
