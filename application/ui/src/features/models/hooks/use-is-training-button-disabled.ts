// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useGetDatasetItems } from '../../../hooks/use-get-dataset-items.hook';

const MIN_NUMBER_OF_ANNOTATED_ITEMS = 3;

export const useIsTrainingButtonDisabled = () => {
    const { data: datasetItems } = useGetDatasetItems({
        annotationStatus: 'reviewed',
    });
    const count = datasetItems?.pagination.total ?? 0;

    return count < MIN_NUMBER_OF_ANNOTATED_ITEMS;
};
