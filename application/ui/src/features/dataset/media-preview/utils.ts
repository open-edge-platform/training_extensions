// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQueryClient } from '@tanstack/react-query';

import type { AnnotationDTO, Media } from '../../../constants/shared-types';
import { loadImageQueryOptions } from '../../annotator/hooks/use-load-image-query.hook';
import {
    segmentAnythingEncodingQueryOptions,
    segmentAnythingWorkerQueryOptions,
} from '../../annotator/tools/segment-anything-tool/use-segment-anything.hook';
import { annotationsQueryOptions } from './api/use-annotations-query';

export const getInitialAnnotations = (isUserReviewed: boolean, annotationsDTO: AnnotationDTO[]): AnnotationDTO[] => {
    return isUserReviewed ? annotationsDTO : [];
};

export const getInitialPredictions = (isUserReviewed: boolean, annotationsDTO: AnnotationDTO[]): AnnotationDTO[] => {
    return isUserReviewed ? [] : annotationsDTO;
};

// When the user navigates to next media, the most expensive data, like the SAM encoding,
// along with image data and annotations, will be already in React Query cache, so the UI will feel smoother
// whenever the user switches image. Unless he/she changes to a random or item. We could also consider
// those cases but I feel like it's overkill. Let's see how this improvement performs and then we can iterate on it.
//
// ensureQueryData will get the data from cache if it's there, or call the queryFn and cache the result if it's not.
// prefetchQuery will fetch the data and cache it
export const prefetchNextMediaItemData = ({
    queryClient,
    projectId,
    getNextMediaItem,
}: {
    queryClient: ReturnType<typeof useQueryClient>;
    projectId: string;
    getNextMediaItem: () => Media | undefined;
}) => {
    const prefetch = async () => {
        const nextItem = getNextMediaItem();

        if (nextItem === undefined) {
            return;
        }

        const nextImage = await queryClient.ensureQueryData(loadImageQueryOptions(projectId, nextItem));

        queryClient.prefetchQuery(annotationsQueryOptions(projectId, nextItem));

        const encoderModel = await queryClient.ensureQueryData(
            segmentAnythingWorkerQueryOptions('SEGMENT_ANYTHING_ENCODER')
        );

        queryClient.prefetchQuery(segmentAnythingEncodingQueryOptions(nextItem, encoderModel, nextImage));
    };

    prefetch();
};
