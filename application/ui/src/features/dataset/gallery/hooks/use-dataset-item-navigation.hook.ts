// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isObject } from 'lodash-es';
import { useNavigate, useParams } from 'react-router';

import { paths } from '../../../../constants/paths';
import { Media } from '../../../../constants/shared-types';
import { useGetDatasetMediaItems } from '../../../../hooks/use-get-dataset-media-items.hook';
import { useProjectIdentifier } from '../../../../hooks/use-project-identifier.hook';

export const useDatasetItemNavigation = () => {
    const navigate = useNavigate();
    const projectId = useProjectIdentifier();
    const { items } = useGetDatasetMediaItems();
    const { datasetItemId } = useParams<{ datasetItemId: string }>();

    const onSelectedMediaItemChange = (item: Media | null) => {
        const route = isObject(item)
            ? paths.project.datasetItem({ projectId, datasetItemId: item.id })
            : paths.project.dataset({ projectId });

        navigate(route);
    };

    return {
        selectedMediaItem: items.find((item) => item.id === datasetItemId) ?? null,
        onSelectedMediaItemChange,
    };
};
