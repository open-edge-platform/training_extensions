// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../api/client';
import type { DatasetRevisionItem, DatasetSubset, Pagination } from '../constants/shared-types';

const DATASET_ITEMS_LIMIT = 20;

interface UseGetDatasetRevisionItemsOptions {
    datasetRevisionId: string;
    subset?: DatasetSubset;
}

export const useGetDatasetRevisionItems = ({ datasetRevisionId, subset }: UseGetDatasetRevisionItemsOptions) => {
    const project_id = useProjectIdentifier();

    const query = subset
        ? { offset: 0, limit: DATASET_ITEMS_LIMIT, subset }
        : { offset: 0, limit: DATASET_ITEMS_LIMIT };

    const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isPending } = $api.useInfiniteQuery(
        'get',
        '/api/projects/{project_id}/dataset_revisions/{dataset_revision_id}/items',
        {
            params: {
                query,
                path: { project_id, dataset_revision_id: datasetRevisionId },
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

    // TODO: OpenAPI spec incorrectly types this as DatasetItem[], but the API actually returns DatasetRevisionItem[]
    // Update this once https://github.com/open-edge-platform/training_extensions/pull/5455 is merged
    const items: DatasetRevisionItem[] = (data?.pages.flatMap((page) => page.items) ?? []) as DatasetRevisionItem[];
    const totalCount = data?.pages[0]?.pagination?.total ?? 0;

    return { items, fetchNextPage, hasNextPage, isFetchingNextPage, isPending, totalCount };
};
