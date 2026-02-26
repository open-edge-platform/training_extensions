// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useNavigate, useParams } from 'react-router';

import { paths } from '../../../../constants/paths';
import { Media } from '../../../../constants/shared-types';
import { useGetDatasetMediaItems } from '../../../../hooks/use-get-dataset-media-items.hook';
import { useProjectIdentifier } from '../../../../hooks/use-project-identifier.hook';
import { isVideo, isVideoFrame } from '../../../../shared/media-item-utils';

export const useSelectDatasetItem = () => {
    const navigate = useNavigate();
    const projectId = useProjectIdentifier();
    const { items } = useGetDatasetMediaItems();
    const { datasetItemId: selectedDatasetItemId } = useParams<{ datasetItemId: string }>();

    const onSelectedMediaItemChange = (item: Media | null) => {
        if (item === null) {
            navigate(paths.project.dataset.index({ projectId }));
            return;
        }

        if (isVideo(item)) {
            navigate(paths.project.dataset.videoFrame({ projectId, datasetItemId: item.id, frameNumber: '0' }));
            return;
        }

        if (isVideoFrame(item)) {
            navigate(
                paths.project.dataset.videoFrame({
                    projectId,
                    datasetItemId: item.id,
                    frameNumber: item.frame_number.toString(),
                })
            );
            return;
        }

        navigate(paths.project.dataset.datasetItem({ projectId, datasetItemId: item.id }));
    };

    return {
        selectedMediaItem: items.find((item) => item.id === selectedDatasetItemId) ?? null,
        onSelectedMediaItemChange,
    };
};
