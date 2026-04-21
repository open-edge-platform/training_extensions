// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { EncodingOutput } from '@geti/smart-tools/segment-anything';
import { queryOptions, skipToken, useQuery } from '@tanstack/react-query';
import { Remote, wrap } from 'comlink';
import { useProject } from 'hooks/api/project.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import type { Media } from '../../../../constants/shared-types';
import { isVideoFrame } from '../../../../shared/media-item-utils';
import { isDetectionTask } from '../../../project/task-type-guards';
import { loadImageQueryOptions } from '../../hooks/use-load-image-query.hook';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import type {
    SegmentAnythingWorkerApi,
    SegmentAnythingWorkerInstance,
} from '../../webworkers/segment-anything.worker.interface';
import { executeWithTimeout } from '../execute-with-timeout';
import { convertToolShapeToGetiShape } from '../utils';
import { InteractiveAnnotationPoint } from './segment-anything.interface';

type SegmentAnythingRemoteInstance = Remote<SegmentAnythingWorkerInstance>;
const SAM_TIMEOUT_MS = 5000;
const SAM_ENCODER_TIMEOUT_MS = 30000;
// Loading the SAM ONNX models (especially the encoder, hundreds of MB) involves a network
// fetch and/or a Chrome Cache Storage hydration that can easily take longer than the decoder
// timing.
const SAM_WORKER_INIT_TIMEOUT_MS = SAM_ENCODER_TIMEOUT_MS;

const getSegmentAnythingWorkerQueryKey = (algorithmType: 'SEGMENT_ANYTHING_DECODER' | 'SEGMENT_ANYTHING_ENCODER') =>
    ['workers', algorithmType] as const;

const segmentAnythingWorkerQueryOptions = (
    algorithmType: 'SEGMENT_ANYTHING_DECODER' | 'SEGMENT_ANYTHING_ENCODER',
    enabled = true
) =>
    queryOptions({
        queryKey: getSegmentAnythingWorkerQueryKey(algorithmType),
        queryFn: async () => {
            const baseWorker = new Worker(new URL('../../webworkers/segment-anything.worker', import.meta.url), {
                type: 'module',
            });
            try {
                const samWorker = wrap<SegmentAnythingWorkerApi>(baseWorker);
                const model = await executeWithTimeout(
                    samWorker.build(),
                    'SAM worker build',
                    SAM_WORKER_INIT_TIMEOUT_MS
                );

                await executeWithTimeout(model.init(algorithmType), 'SAM worker init', SAM_WORKER_INIT_TIMEOUT_MS);

                return model;
            } catch (error) {
                baseWorker.terminate();
                throw error;
            }
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
    model: SegmentAnythingRemoteInstance | undefined,
    image: ImageData,
    enabled = true
) =>
    queryOptions({
        queryKey: getSegmentAnythingEncodingQueryKey(mediaItem),
        queryFn: async () => {
            if (model === undefined) {
                throw new Error('Model not yet initialized');
            }

            return executeWithTimeout(model.processEncoder(image), 'SAM encoder', SAM_ENCODER_TIMEOUT_MS);
        },
        staleTime: Infinity,
        gcTime: 3600 * 15,
        enabled,
    });

export const useSegmentAnythingWorker = (
    algorithmType: 'SEGMENT_ANYTHING_DECODER' | 'SEGMENT_ANYTHING_ENCODER',
    enabled = true
) => {
    return useQuery(segmentAnythingWorkerQueryOptions(algorithmType, enabled));
};

const useEncodingQuery = (
    model: SegmentAnythingRemoteInstance | undefined,
    mediaItem: Media | undefined,
    image: ImageData | undefined,
    isImageReady: boolean
) => {
    const isEnabled = model !== undefined && mediaItem !== undefined && image !== undefined && isImageReady;

    return useQuery(
        mediaItem !== undefined && image !== undefined
            ? segmentAnythingEncodingQueryOptions(mediaItem, model, image, isEnabled)
            : {
                  queryKey: ['segment-anything-model', 'encoding', 'disabled'],
                  queryFn: skipToken,
              }
    );
};

const useDecoderOutputType = () => {
    const { data } = useProject();

    if (isDetectionTask(data.task.task_type)) {
        return 'rect';
    }

    return 'polygon';
};

const useDecodingFn = (model: SegmentAnythingRemoteInstance | undefined, encoding: EncodingOutput | undefined) => {
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

        const { shapes } = await executeWithTimeout(
            model.processDecoder(encoding, {
                points,
                boxes: [],
                ouputConfig: {
                    type: decoderOutput,
                },
                image: undefined,
            }),
            'SAM decoder',
            SAM_TIMEOUT_MS
        );

        return shapes.map(convertToolShapeToGetiShape);
    };
};

type SegmentAnythingModelOptions = {
    nextMediaItem?: Media;
};

export const useSegmentAnythingModel = ({ nextMediaItem }: SegmentAnythingModelOptions = {}) => {
    const encoderWorkerQuery = useSegmentAnythingWorker('SEGMENT_ANYTHING_ENCODER');
    const decoderWorkerQuery = useSegmentAnythingWorker('SEGMENT_ANYTHING_DECODER');
    const encoderModel = encoderWorkerQuery.data;
    const decoderModel = decoderWorkerQuery.data;
    const hasWorkerError = encoderWorkerQuery.isError || decoderWorkerQuery.isError;
    const isLoadingWorkers = encoderWorkerQuery.isLoading || decoderWorkerQuery.isLoading;
    const projectId = useProjectIdentifier();

    const { mediaItem, image, isImageReady } = useSelectedMediaItem();
    const nextImageQuery = useQuery({
        ...loadImageQueryOptions(projectId, nextMediaItem ?? mediaItem),
        enabled: nextMediaItem !== undefined,
    });

    // First we get the encoding for the CURRENT image
    const encodingQuery = useEncodingQuery(encoderModel, mediaItem, image, isImageReady);

    // At the same time we start prefetching the encoding for the NEXT image,
    // so when the user moves to the next media item the decoding will be faster.
    // We don't need to get the decoding query result for the next image, we just want to cache the encoding result.
    const canPrefetch = nextImageQuery.isSuccess && !encodingQuery.isFetching;
    useEncodingQuery(encoderModel, nextMediaItem, nextImageQuery.data, canPrefetch);

    const decodingQueryFn = useDecodingFn(decoderModel, encodingQuery.data);

    const isLoading = !hasWorkerError && (isLoadingWorkers || encodingQuery.isLoading);
    const isProcessing = encodingQuery.isFetching;
    const isError = hasWorkerError || encodingQuery.isError;
    const error = encoderWorkerQuery.error ?? decoderWorkerQuery.error ?? encodingQuery.error;

    return {
        isLoading,
        isProcessing,
        isError,
        error,
        encodingQuery,
        decodingQueryFn,
    };
};
