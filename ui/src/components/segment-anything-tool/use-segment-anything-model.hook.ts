// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

import { EncodingOutput, SegmentAnythingModel } from '@geti/smart-tools/segment-anything';
import { useQuery } from '@tanstack/react-query';
import { Remote } from 'comlink';

import { AlgorithmType } from '../../webworkers/algorithm.interface';
import { useLoadAIWebworker } from '../../webworkers/use-load-ai-webworker.hook';
import { convertToolShapeToGetiShape } from '../utils';
import { InteractiveAnnotationPoint } from './segment-anything.interface';

const useDecodingFn = (model: Remote<SegmentAnythingModel> | undefined, encoding: EncodingOutput | undefined) => {
    const shapeType = 'polygon';

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
                type: shapeType,
            },
            image: undefined,
        });

        return shapes.map(convertToolShapeToGetiShape);
    };
};

const useEncodingQuery = (model: Remote<SegmentAnythingModel> | undefined, mediaItem: unknown) => {
    return useQuery({
        queryKey: ['segment-anything-model', 'encoding', mediaItem?.identifier],
        queryFn: async () => {
            if (model === undefined) {
                throw new Error('Model not yet initialized');
            }

            if (mediaItem === undefined) {
                throw new Error('Media item not selected');
            }

            return await model.processEncoder(mediaItem.image);
        },
        staleTime: Infinity,
        gcTime: 3600 * 15,
        enabled: model !== undefined && mediaItem !== undefined,
    });
};

const useSegmentAnythingWorker = (
    algorithmType: AlgorithmType.SEGMENT_ANYTHING_DECODER | AlgorithmType.SEGMENT_ANYTHING_ENCODER
) => {
    const { worker } = useLoadAIWebworker(algorithmType);

    const modelRef = useRef<Remote<SegmentAnythingModel>>(undefined);
    const [modelIsLoading, setModelIsLoading] = useState(false);

    useEffect(() => {
        const loadWorker = async () => {
            setModelIsLoading(true);

            if (worker) {
                const model = worker;

                await model.init(algorithmType);

                modelRef.current = model;
            }

            setModelIsLoading(false);
        };

        if (worker && modelRef.current === undefined && !modelIsLoading) {
            loadWorker();
        }
    }, [worker, modelIsLoading, algorithmType]);

    return modelRef.current;
};

export const useSegmentAnythingModel = () => {
    const encoderModel = useSegmentAnythingWorker(AlgorithmType.SEGMENT_ANYTHING_ENCODER);
    const decoderModel = useSegmentAnythingWorker(AlgorithmType.SEGMENT_ANYTHING_DECODER);
    const isLoading = encoderModel === undefined || decoderModel === undefined;

    const { selectedMediaItem } = useSelectedMediaItem();
    const encodingQuery = useEncodingQuery(encoderModel, selectedMediaItem);
    const decodingQueryFn = useDecodingFn(decoderModel, encodingQuery.data);

    const nextSelectedMediaItem = useNextMediaItemWithImage();
    useEncodingQuery(encoderModel, encodingQuery.isFetching ? undefined : nextSelectedMediaItem);

    return { isLoading, encodingQuery, decodingQueryFn };
};
