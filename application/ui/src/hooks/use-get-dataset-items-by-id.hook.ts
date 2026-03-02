// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { useGetDatasetItems } from './use-get-dataset-items.hook';

export const useGetDatasetItemsById = (options?: Parameters<typeof useGetDatasetItems>[0]) => {
    const { data } = useGetDatasetItems(options);

    const datasetItemsById = useMemo(() => {
        const datasetItems = data?.items ?? [];

        return new Map(datasetItems.map(({ id, user_reviewed }) => [id, user_reviewed]));
    }, [data?.items]);

    return { datasetItemsById };
};
