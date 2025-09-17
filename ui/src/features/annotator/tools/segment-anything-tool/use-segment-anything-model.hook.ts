// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

import { EncodingOutput, SegmentAnythingModel } from '@geti/smart-tools/segment-anything';
import { useQuery } from '@tanstack/react-query';
import { Remote } from 'comlink';

import { useSegmentAnythingWorkerQuery } from '../../hooks/use-segment-anything.hook';
import { convertToolShapeToGetiShape } from '../utils';
import { InteractiveAnnotationPoint } from './segment-anything.interface';

const selectedMediaItem = {
    identifier: 'id',
    image: new ImageData(100, 100),
};

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
        queryKey: ['segment-anything-model', 'encoding', selectedMediaItem?.identifier],
        queryFn: async () => {
            if (model === undefined) {
                throw new Error('Model not yet initialized');
            }

            if (mediaItem === undefined) {
                throw new Error('Media item not selected');
            }

            return await model.processEncoder(selectedMediaItem.image);
        },
        staleTime: Infinity,
        gcTime: 3600 * 15,
        enabled: model !== undefined && mediaItem !== undefined,
    });
};

const useSegmentAnythingWorker = (algorithmType: 'SEGMENT_ANYTHING_DECODER' | 'SEGMENT_ANYTHING_ENCODER') => {
    const worker = useSegmentAnythingWorkerQuery(algorithmType);

    const [model, setModel] = useState<Remote<SegmentAnythingModel> | undefined>(undefined);
    const [modelIsLoading, setModelIsLoading] = useState(false);

    useEffect(() => {
        const loadWorker = async () => {
            setModelIsLoading(true);

            if (worker) {
                console.log(worker);
                const modelInstance = worker.data;

                console.log('modelInstance: ', modelInstance);
                if (modelInstance) {
                    await modelInstance.init(algorithmType);

                    setModel(modelInstance);
                }
            }

            setModelIsLoading(false);
        };

        if (worker && model === undefined && !modelIsLoading) {
            console.log('loading worker');
            loadWorker();
        }
    }, [worker, modelIsLoading, algorithmType, model]);

    return model;
};

export const useSegmentAnythingModel = () => {
    const encoderModel = useSegmentAnythingWorker('SEGMENT_ANYTHING_ENCODER');
    const decoderModel = useSegmentAnythingWorker('SEGMENT_ANYTHING_DECODER');
    const isLoading = encoderModel === undefined || decoderModel === undefined;

    const encodingQuery = useEncodingQuery(encoderModel, selectedMediaItem);
    const decodingQueryFn = useDecodingFn(decoderModel, encodingQuery.data);

    useEncodingQuery(encoderModel, encodingQuery.isFetching ? undefined : selectedMediaItem);

    return { isLoading, encodingQuery, decodingQueryFn };
};
