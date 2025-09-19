// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

import { EncodingOutput, SegmentAnythingModel } from '@geti/smart-tools/segment-anything';
import { useQuery } from '@tanstack/react-query';
import { Remote, wrap } from 'comlink';

import { useAnnotator } from '../../annotator-provider.component';
import { MediaItem } from '../../types';
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
    selectedMediaItem: Pick<MediaItem, 'id' | 'width' | 'height'> | undefined
) => {
    return useQuery({
        queryKey: ['segment-anything-model', 'encoding', selectedMediaItem?.id],
        queryFn: async () => {
            if (model === undefined) {
                throw new Error('Model not yet initialized');
            }

            if (selectedMediaItem === undefined) {
                throw new Error('Media item not selected');
            }

            return await model.processEncoder(new ImageData(selectedMediaItem.width, selectedMediaItem.height));
        },
        staleTime: Infinity,
        gcTime: 3600 * 15, // WIP
        enabled: model !== undefined && selectedMediaItem !== undefined,
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
    const isLoading = encoderModel === undefined || decoderModel === undefined;

    const { mediaItem } = useAnnotator();
    const encodingQuery = useEncodingQuery(encoderModel, mediaItem);
    const decodingQueryFn = useDecodingFn(decoderModel, encodingQuery.data);

    return { isLoading, encodingQuery, decodingQueryFn };
};
