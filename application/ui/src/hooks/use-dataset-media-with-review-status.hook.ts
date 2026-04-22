// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isEmpty } from 'lodash';

import type { DatasetItemAnnotationStatus } from '../constants/shared-types';
import { useGetDatasetItemsById } from './use-get-dataset-items-by-id.hook';
import { useGetDatasetMediaItems } from './use-get-dataset-media-items.hook';
import { useLabelsSearchParams } from './use-labels-search-params.hook';

interface UseDatasetMediaWithReviewStatusOptions {
    annotationStatus?: DatasetItemAnnotationStatus;
}

export const useDatasetMediaWithReviewStatus = ({ annotationStatus }: UseDatasetMediaWithReviewStatusOptions) => {
    const { selectedLabelIds } = useLabelsSearchParams();

    const mediaItemsResponse = useGetDatasetMediaItems({
        annotationStatus,
        labels: isEmpty(selectedLabelIds) ? undefined : selectedLabelIds,
    });

    const datasetItemsResponse = useGetDatasetItemsById({ annotationStatus });

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
        fetchNextPage,
        isMediaItemReviewedById,
    };
};
