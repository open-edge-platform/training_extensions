// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { DatasetItemAnnotationStatus } from '../constants/shared-types';
import { useGetDatasetItemsById } from './use-get-dataset-items-by-id.hook';
import { useGetDatasetMediaItems } from './use-get-dataset-media-items.hook';

interface UseDatasetMediaWithReviewStatusOptions {
    annotationStatus?: DatasetItemAnnotationStatus;
}

export const useDatasetMediaWithReviewStatus = ({ annotationStatus }: UseDatasetMediaWithReviewStatusOptions) => {
    const mediaItemsResponse = useGetDatasetMediaItems({ annotationStatus });
    const datasetItemsResponse = useGetDatasetItemsById({ annotationStatus });

    const fetchNextPage = () => {
        if (mediaItemsResponse.hasNextPage && !mediaItemsResponse.isFetchingNextPage) {
            mediaItemsResponse.fetchNextPage();
        }

        if (datasetItemsResponse.hasNextPage && !datasetItemsResponse.isFetchingNextPage) {
            datasetItemsResponse.fetchNextPage();
        }
    };

    const isUserReviewed = (mediaItemId: string) => {
        return datasetItemsResponse.reviewStatus.get(mediaItemId) ?? false;
    };

    return {
        items: mediaItemsResponse.items,
        isPending: mediaItemsResponse.isPending || datasetItemsResponse.isPending,
        isFetchingNextPage: mediaItemsResponse.isFetchingNextPage || datasetItemsResponse.isFetchingNextPage,
        fetchNextPage,
        isUserReviewed,
    };
};
