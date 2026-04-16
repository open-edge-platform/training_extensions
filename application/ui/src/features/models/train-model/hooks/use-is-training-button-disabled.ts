// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useGetDatasetItems } from 'hooks/use-get-dataset-items.hook';

const MIN_NUMBER_OF_ANNOTATED_ITEMS = 3;

const REVIEWED_DATASET_ITEMS_OPTIONS = {
    annotationStatus: 'reviewed',
} as const;

export const useIsTrainingButtonDisabled = () => {
    const { totalCount } = useGetDatasetItems(REVIEWED_DATASET_ITEMS_OPTIONS);

    return totalCount < MIN_NUMBER_OF_ANNOTATED_ITEMS;
};
