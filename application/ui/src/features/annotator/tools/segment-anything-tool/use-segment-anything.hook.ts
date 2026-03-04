// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { EncodingOutput } from '@geti/smart-tools/segment-anything';
import { queryOptions, useQuery } from '@tanstack/react-query';
import { Remote, wrap } from 'comlink';
import { useProject } from 'hooks/api/project.hook';

import type { Media } from '../../../../constants/shared-types';
import { isVideoFrame } from '../../../../shared/media-item-utils';
import { isDetectionTask } from '../../../project/task-type-guards';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import type {
    SegmentAnythingWorkerApi,
    SegmentAnythingWorkerModel,
} from '../../webworkers/segment-anything.worker.interface';
import { convertToolShapeToGetiShape } from '../utils';
import { InteractiveAnnotationPoint } from './segment-anything.interface';

type SegmentAnythingRemoteModel = Remote<SegmentAnythingWorkerModel>;

export const getSegmentAnythingWorkerQueryKey = (
    algorithmType: 'SEGMENT_ANYTHING_DECODER' | 'SEGMENT_ANYTHING_ENCODER'
) => ['workers', algorithmType] as const;

export const segmentAnythingWorkerQueryOptions = (
    algorithmType: 'SEGMENT_ANYTHING_DECODER' | 'SEGMENT_ANYTHING_ENCODER',
    enabled = true
) =>
    queryOptions({
        queryKey: getSegmentAnythingWorkerQueryKey(algorithmType),
        queryFn: async () => {
            const baseWorker = new Worker(new URL('../../webworkers/segment-anything.worker', import.meta.url), {
                type: 'module',
            });
            const samWorker = wrap<SegmentAnythingWorkerApi>(baseWorker);
            const model = await samWorker.build();

            await model.init(algorithmType);

            return model;
        },
        staleTime: Infinity,
        enabled,
    });

const getSegmentAnythingEncodingQueryKey = (mediaItem: Media) => {
    return isVideoFrame(mediaItem)
        ? ['segment-anything-model', 'encoding', mediaItem.id, mediaItem.frame_number]
        : ['segment-anything-model', 'encoding', mediaItem.id];
};

export const segmentAnythingEncodingQueryOptions = (
    mediaItem: Media,
    model: SegmentAnythingRemoteModel | undefined,
    image: ImageData,
    enabled = true
) =>
    queryOptions({
        queryKey: getSegmentAnythingEncodingQueryKey(mediaItem),
        queryFn: async () => {
            if (model === undefined) {
                throw new Error('Model not yet initialized');
            }

            return model.processEncoder(image);
        },
        staleTime: Infinity,
        gcTime: 3600 * 15,
        enabled,
    });

const useSegmentAnythingWorker = (
    algorithmType: 'SEGMENT_ANYTHING_DECODER' | 'SEGMENT_ANYTHING_ENCODER',
    enabled = true
) => {
    const { data } = useQuery(segmentAnythingWorkerQueryOptions(algorithmType, enabled));

    return data;
};

export const usePreloadSAMWorkers = (enabled = true) => {
    useSegmentAnythingWorker('SEGMENT_ANYTHING_ENCODER', enabled);
    useSegmentAnythingWorker('SEGMENT_ANYTHING_DECODER', enabled);
};

const useEncodingQuery = (
    model: SegmentAnythingRemoteModel | undefined,
    mediaItem: Media | undefined,
    image: ImageData | undefined,
    isImageReady: boolean
) => {
    const isEnabled = model !== undefined && mediaItem !== undefined && image !== undefined && isImageReady;

    return useQuery({
        queryKey:
            mediaItem === undefined
                ? ['segment-anything-model', 'encoding', 'disabled']
                : getSegmentAnythingEncodingQueryKey(mediaItem),
        queryFn: async () => {
            if (model === undefined || image === undefined) {
                throw new Error('Model not yet initialized');
            }

            return model.processEncoder(image);
        },
        staleTime: Infinity,
        gcTime: 3600 * 15,
        enabled: isEnabled,
    });
};

const useDecoderOutputType = () => {
    const { data } = useProject();

    if (isDetectionTask(data.task.task_type)) {
        return 'rect';
    }

    return 'polygon';
};

const useDecodingFn = (model: SegmentAnythingRemoteModel | undefined, encoding: EncodingOutput | undefined) => {
    const decoderOutput = useDecoderOutputType();

    // TODO: look into returning a new "decoder model" instance that already has the encoding data
    // stored in memory, to reduce  memory usage
    return async (points: InteractiveAnnotationPoint[]) => {
        if (points.length === 0) {
            return [];
        }

        if (model === undefined) {
            return [];
        }

        if (encoding === undefined) {
            return [];
        }

        const { shapes } = await model.processDecoder(encoding, {
            points,
            boxes: [],
            ouputConfig: {
                type: decoderOutput,
            },
            image: undefined,
        });

        return shapes.map(convertToolShapeToGetiShape);
    };
};

type SegmentAnythingModelOptions = {
    nextMediaItem?: Media;
    nextImage?: ImageData;
    isNextImageReady?: boolean;
};

export const useSegmentAnythingModel = ({
    nextMediaItem,
    nextImage,
    isNextImageReady = false,
}: SegmentAnythingModelOptions = {}) => {
    const encoderModel = useSegmentAnythingWorker('SEGMENT_ANYTHING_ENCODER');
    const decoderModel = useSegmentAnythingWorker('SEGMENT_ANYTHING_DECODER');
    const isLoadingWorkers = encoderModel === undefined || decoderModel === undefined;

    const { mediaItem, image, isImageReady } = useSelectedMediaItem();

    // First we get the encoding for the CURRENT image
    const encodingQuery = useEncodingQuery(encoderModel, mediaItem, image, isImageReady);
    // At the same time we start prefetching the encoding for the NEXT image,
    // so when the user moves to the next media item the decoding will be faster.
    // We don't need to get the decoding query result for the next image, we just want to cache the encoding result.
    useEncodingQuery(encoderModel, nextMediaItem, nextImage, isNextImageReady);

    const decodingQueryFn = useDecodingFn(decoderModel, encodingQuery.data);

    const isLoading = isLoadingWorkers || encodingQuery.isLoading;
    const isProcessing = encodingQuery.isFetching;

    return {
        isLoading,
        isProcessing,
        encodingQuery,
        decodingQueryFn,
    };
};
