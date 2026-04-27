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
const SAM_DECODER_TIMEOUT_MS = 20000;
const SAM_ENCODER_TIMEOUT_MS = 30000;
const SAM_WORKER_BUILD_TIMEOUT_MS = 10000;
const SAM_WORKER_INIT_TIMEOUT_MS = SAM_ENCODER_TIMEOUT_MS;
// How long an unobserved encoding (a large Float32 tensor per image) is kept in
// the query cache before being garbage-collected. Short on purpose: encodings
// are heavy and are cheap-ish to recompute, so we favor memory over perf.
const SAM_ENCODING_GC_TIME_MS = 60_000;

// A single shared worker hosts BOTH the encoder and decoder ONNX sessions.
// Spawning two workers used to double the OpenCV + ONNX Runtime WASM footprint
// for no functional gain (encoder/decoder always run sequentially anyway).
const segmentAnythingWorkerQueryOptions = (enabled = true) =>
    queryOptions<{ worker: Worker; instance: SegmentAnythingRemoteInstance }>({
        queryKey: ['workers', 'SEGMENT_ANYTHING'],
        queryFn: async ({ signal }) => {
            const worker = new Worker(new URL('../../webworkers/segment-anything.worker', import.meta.url), {
                type: 'module',
            });
            // Terminate the worker if the query is cancelled (e.g. annotator unmounts)
            // before build/init resolve, so we don't leak the in-flight worker.
            signal.addEventListener('abort', worker.terminate, { once: true });

            try {
                const samWorker = wrap<SegmentAnythingWorkerApi>(worker);
                const instance = await executeWithTimeout(
                    samWorker.build(),
                    'SAM worker build',
                    SAM_WORKER_BUILD_TIMEOUT_MS
                );

                // Initialize encoder and decoder sessions in parallel inside the same worker.
                await executeWithTimeout(
                    Promise.all([instance.init('SEGMENT_ANYTHING_ENCODER'), instance.init('SEGMENT_ANYTHING_DECODER')]),
                    'SAM worker init',
                    SAM_WORKER_INIT_TIMEOUT_MS
                );

                if (signal.aborted) {
                    throw signal.reason;
                }

                return { worker, instance };
            } catch (error) {
                worker.terminate();

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
        gcTime: SAM_ENCODING_GC_TIME_MS,
        enabled,
    });

export const useSegmentAnythingWorker = (enabled = true) => {
    return useQuery({
        ...segmentAnythingWorkerQueryOptions(enabled),
        select: (data) => data.instance,
    });
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
            SAM_DECODER_TIMEOUT_MS
        );

        return shapes.map(convertToolShapeToGetiShape);
    };
};

type SegmentAnythingModelOptions = {
    nextMediaItem?: Media;
};

export const useSegmentAnythingModel = ({ nextMediaItem }: SegmentAnythingModelOptions = {}) => {
    const workerQuery = useSegmentAnythingWorker();
    const model = workerQuery.data;
    const hasWorkerError = workerQuery.isError;
    const isLoadingWorkers = workerQuery.isLoading;
    const projectId = useProjectIdentifier();

    const { mediaItem, image, isImageReady } = useSelectedMediaItem();
    const nextImageQuery = useQuery({
        ...loadImageQueryOptions(projectId, nextMediaItem ?? mediaItem),
        enabled: nextMediaItem !== undefined,
    });

    // First we get the encoding for the CURRENT image
    const encodingQuery = useEncodingQuery(model, mediaItem, image, isImageReady);

    // At the same time we start prefetching the encoding for the NEXT image,
    // so when the user moves to the next media item the decoding will be faster.
    // We don't need to get the decoding query result for the next image, we just want to cache the encoding result.
    const canPrefetch = nextImageQuery.isSuccess && !encodingQuery.isFetching;
    useEncodingQuery(model, nextMediaItem, nextImageQuery.data, canPrefetch);

    const decodingQueryFn = useDecodingFn(model, encodingQuery.data);

    const isLoading = !hasWorkerError && (isLoadingWorkers || encodingQuery.isLoading);
    const isProcessing = encodingQuery.isFetching;
    const isError = hasWorkerError || encodingQuery.isError;
    const error = workerQuery.error ?? encodingQuery.error;

    return {
        isLoading,
        isProcessing,
        isError,
        error,
        encodingQuery,
        decodingQueryFn,
    };
};
