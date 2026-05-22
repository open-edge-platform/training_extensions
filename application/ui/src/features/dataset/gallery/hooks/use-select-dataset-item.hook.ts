// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLocation, useNavigate, useParams } from 'react-router';

import { paths } from '../../../../constants/paths';
import { Media } from '../../../../constants/shared-types';
import { useGetDatasetMediaItems } from '../../../../hooks/use-get-dataset-media-items.hook';
import { useProjectIdentifier } from '../../../../hooks/use-project-identifier.hook';
import { isVideo, isVideoFrame } from '../../../../shared/media-item-utils';

export const useSelectDatasetItem = () => {
    const navigate = useNavigate();
    const { search } = useLocation();
    const projectId = useProjectIdentifier();
    const { items } = useGetDatasetMediaItems();
    const { datasetItemId: selectedDatasetItemId } = useParams<{ datasetItemId: string }>();

    const onSelectedMediaItemChange = (item: Media | null) => {
        if (item === null) {
            navigate({ pathname: paths.project.dataset.index({ projectId }), search });
            return;
        }

        if (isVideo(item)) {
            navigate({
                pathname: paths.project.dataset.item.frame({ projectId, datasetItemId: item.id, frameNumber: '0' }),
                search,
            });
            return;
        }

        if (isVideoFrame(item)) {
            navigate({
                pathname: paths.project.dataset.item.frame({
                    projectId,
                    datasetItemId: item.id,
                    frameNumber: item.frame_number.toString(),
                }),
                search,
            });
            return;
        }

        navigate({ pathname: paths.project.dataset.item.index({ projectId, datasetItemId: item.id }), search });
    };

    return {
        selectedMediaItem: items.find((item) => item.id === selectedDatasetItemId) ?? null,
        onSelectedMediaItemChange,
    };
};
