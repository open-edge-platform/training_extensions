// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../api/client';

const useGetDatasetItems = () => {
    const projectId = useProjectIdentifier();
    return $api.useQuery('get', '/api/projects/{project_id}/dataset/items', {
        params: {
            path: {
                project_id: projectId,
            },
            query: {
                limit: 5,
                annotation_status: 'reviewed',
            },
        },
    });
};

const MIN_NUMBER_OF_ANNOTATED_ITEMS = 3;

export const useIsTrainingButtonDisabled = () => {
    const { data: datasetItems } = useGetDatasetItems();
    const count = datasetItems?.items.length ?? 0;

    return count < MIN_NUMBER_OF_ANNOTATED_ITEMS;
};
