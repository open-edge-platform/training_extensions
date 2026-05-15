// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import type { Media } from '../../../constants/shared-types';
import { isVideoFrame } from '../../../shared/media-item-utils';
import { mediaPredictionsQueryOptions } from '../../annotator/api/use-media-predictions';
import { usePredictionSetup } from '../../annotator/predictions-setup-provider.component';
import { useNextMediaItem } from './utils';

export const useNextPredictionPrefetch = (currentMediaItem: Media, allMediaItems: Media[], isEnabled: boolean) => {
    const queryClient = useQueryClient();
    const projectId = useProjectIdentifier();
    const { selectedModel } = usePredictionSetup();
    const nextMediaItem = useNextMediaItem(currentMediaItem, allMediaItems);

    if (isEnabled === false || nextMediaItem === undefined) {
        return;
    }

    const range = isVideoFrame(nextMediaItem)
        ? {
              stride: nextMediaItem.frame_stride,
              end_frame: nextMediaItem.frame_number,
              start_frame: nextMediaItem.frame_number,
          }
        : null;

    queryClient.prefetchQuery(
        mediaPredictionsQueryOptions({
            projectId,
            selectedModel,
            mediaId: nextMediaItem.id,
            range,
        })
    );
};
