// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

import { EncodingOutput, SegmentAnythingModel } from '@geti/smart-tools/segment-anything';
import { useQuery } from '@tanstack/react-query';
import { Remote, wrap } from 'comlink';

import { useAnnotator } from '../../annotator-provider.component';
import { DatasetItem } from '../../types';
import { convertToolShapeToGetiShape } from '../utils';
import { InteractiveAnnotationPoint } from './segment-anything.interface';

const useSegmentAnythingWorker = (algorithmType: 'SEGMENT_ANYTHING_DECODER' | 'SEGMENT_ANYTHING_ENCODER') => {
    const { data } = useQuery<Remote<SegmentAnythingModel>>({
        queryKey: ['workers', algorithmType],
        queryFn: async () => {
            const baseWorker = new Worker(new URL('../../webworkers/segment-anything.worker', import.meta.url), {
                type: 'module',
            });
            const samWorker = wrap(baseWorker);

            // @ts-expect-error build exists on every worker
            return samWorker.build();
        },
        staleTime: Infinity,
    });

    const modelRef = useRef<Remote<SegmentAnythingModel>>(undefined);
    const [modelIsLoading, setModelIsLoading] = useState(false);

    useEffect(() => {
        const loadWorker = async () => {
            setModelIsLoading(true);

            if (data) {
                const model = data;

                await model.init(algorithmType);

                modelRef.current = model;
            }

            setModelIsLoading(false);
        };

        if (data && modelRef.current === undefined && !modelIsLoading) {
            loadWorker();
        }
    }, [data, modelIsLoading, algorithmType]);

    return modelRef.current;
};

const useEncodingQuery = (
    model: Remote<SegmentAnythingModel> | undefined,
    mediaItem: DatasetItem,
    image: ImageData
) => {
    return useQuery({
        queryKey: ['segment-anything-model', 'encoding', mediaItem?.id],
        queryFn: async () => {
            if (model === undefined) {
                throw new Error('Model not yet initialized');
            }

            if (image === undefined) {
                throw new Error('Image not available');
            }

            return await model.processEncoder(image);
        },
        staleTime: Infinity,
        gcTime: 3600 * 15,
        enabled: model !== undefined && mediaItem !== undefined,
    });
};

const useDecodingFn = (model: Remote<SegmentAnythingModel> | undefined, encoding: EncodingOutput | undefined) => {
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
                type: 'polygon',
            },
            image: undefined,
        });

        return shapes.map(convertToolShapeToGetiShape);
    };
};

export const useSegmentAnythingModel = () => {
    const encoderModel = useSegmentAnythingWorker('SEGMENT_ANYTHING_ENCODER');
    const decoderModel = useSegmentAnythingWorker('SEGMENT_ANYTHING_DECODER');
    const isLoadingWorkers = encoderModel === undefined || decoderModel === undefined;

    const { mediaItem, image } = useAnnotator();
    const encodingQuery = useEncodingQuery(encoderModel, mediaItem, image);
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
