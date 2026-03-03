// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { DatasetStatistics } from '../../../../components/dataset-statistics/dataset-statistics.component';
import { useGetDatasetItems } from '../../../../hooks/use-get-dataset-items.hook';

export const MainDatasetStatistics = () => {
    const { data: annotatedItems } = useGetDatasetItems({ annotationStatus: 'reviewed', limit: 1 });
    const { data: mediaItems } = useGetDatasetItems({ limit: 1 });

    const totalMediaItems = mediaItems?.pagination.total ?? 0;
    const totalAnnotatedItems = annotatedItems?.pagination.total ?? 0;

    return <DatasetStatistics totalMediaItems={totalMediaItems} totalAnnotatedItems={totalAnnotatedItems} />;
};
