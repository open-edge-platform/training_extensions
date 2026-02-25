// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';
import { DatasetStatistics } from '../../../../components/dataset-statistics/dataset-statistics.component';

export const MainDatasetStatistics = () => {
    const projectId = useProjectIdentifier();

    const { data: annotatedItems } = $api.useQuery('get', '/api/projects/{project_id}/dataset/items', {
        params: {
            path: { project_id: projectId },
            query: { limit: 1, annotation_status: 'reviewed' },
        },
    });

    const { data: mediaItems } = $api.useQuery('get', '/api/projects/{project_id}/dataset/items', {
        params: { path: { project_id: projectId } },
    });

    const totalMediaItems = mediaItems?.pagination.total ?? 0;
    const totalAnnotatedItems = annotatedItems?.pagination.total ?? 0;

    return <DatasetStatistics totalMediaItems={totalMediaItems} totalAnnotatedItems={totalAnnotatedItems} />;
};
