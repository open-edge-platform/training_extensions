// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { useGetDatasetItems } from './use-get-dataset-items.hook';

type UseGetDatasetItemsByIdOptions = {
    limit?: number;
};

export const useGetDatasetItemsById = (options?: UseGetDatasetItemsByIdOptions) => {
    const { data } = useGetDatasetItems({ limit: options?.limit });

    const datasetItemsById = useMemo(() => {
        const datasetItems = data?.items ?? [];

        return new Map(datasetItems.map(({ id, user_reviewed }) => [id, user_reviewed]));
    }, [data?.items]);

    return { datasetItemsById };
};
