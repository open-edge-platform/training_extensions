// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePrefetchQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import type { Media } from '../../../constants/shared-types';
import { isVideoFrame } from '../../../shared/media-item-utils';
import { mediaPredictionsQueryOptions } from '../../annotator/api/use-media-predictions';
import { usePredictionSetup } from '../../annotator/predictions-setup-provider.component';
import { useNextMediaItem } from './utils';

export const useNextPredictionPrefetch = (currentMediaItem: Media, allMediaItems: Media[]) => {
    const projectId = useProjectIdentifier();
    const { selectedModel } = usePredictionSetup();
    const nextMediaItem = useNextMediaItem(currentMediaItem, allMediaItems);

    const mediaToPrefetch = nextMediaItem ?? currentMediaItem;
    const range = isVideoFrame(mediaToPrefetch)
        ? {
              start_frame: mediaToPrefetch.frame_number,
              end_frame: mediaToPrefetch.frame_number,
              stride: mediaToPrefetch.frame_stride,
          }
        : null;

    usePrefetchQuery({
        ...mediaPredictionsQueryOptions({
            projectId,
            selectedModel,
            mediaId: mediaToPrefetch.id,
            range,
        }),
    });
};
