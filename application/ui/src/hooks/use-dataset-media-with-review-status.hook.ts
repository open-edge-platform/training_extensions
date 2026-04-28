// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isEmpty } from 'lodash-es';

import { useDatasetFiltersSearchParams } from './use-dataset-filters-search-params.hook';
import { useGetDatasetItemsById } from './use-get-dataset-items-by-id.hook';
import { useGetDatasetMediaItems } from './use-get-dataset-media-items.hook';

export const useDatasetMediaWithReviewStatus = () => {
    const { selectedLabelIds, annotationStatus } = useDatasetFiltersSearchParams();

    const mediaItemsResponse = useGetDatasetMediaItems({
        annotationStatus: annotationStatus ?? undefined,
        labelIds: isEmpty(selectedLabelIds) ? undefined : selectedLabelIds,
    });

    const datasetItemsResponse = useGetDatasetItemsById({ annotationStatus: annotationStatus ?? undefined });

    const fetchNextPage = () => {
        if (mediaItemsResponse.hasNextPage && !mediaItemsResponse.isFetchingNextPage) {
            mediaItemsResponse.fetchNextPage();
        }

        if (datasetItemsResponse.hasNextPage && !datasetItemsResponse.isFetchingNextPage) {
            datasetItemsResponse.fetchNextPage();
        }
    };

    const isMediaItemReviewedById = (mediaItemId: string) => {
        return datasetItemsResponse.reviewStatus.get(mediaItemId) ?? false;
    };

    return {
        items: mediaItemsResponse.items,
        isPending: mediaItemsResponse.isPending || datasetItemsResponse.isPending,
        isFetchingNextPage: mediaItemsResponse.isFetchingNextPage || datasetItemsResponse.isFetchingNextPage,
        totalCount: mediaItemsResponse.totalCount,
        fetchNextPage,
        isMediaItemReviewedById,
    };
};
