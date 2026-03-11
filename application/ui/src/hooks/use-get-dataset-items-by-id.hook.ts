// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import type { DatasetItemAnnotationStatus } from '../constants/shared-types';
import { useGetDatasetItems } from './use-get-dataset-items.hook';

type UseGetDatasetItemsByIdOptions = {
    limit: number;
    annotationStatus?: DatasetItemAnnotationStatus;
};

export const useGetDatasetItemsById = ({ limit, annotationStatus }: UseGetDatasetItemsByIdOptions) => {
    const { data } = useGetDatasetItems({ limit, annotationStatus });

    const datasetItemsById = useMemo(() => {
        const datasetItems = data?.items ?? [];

        return new Map(datasetItems.map(({ id, user_reviewed }) => [id, user_reviewed]));
    }, [data?.items]);

    return { datasetItemsById };
};
