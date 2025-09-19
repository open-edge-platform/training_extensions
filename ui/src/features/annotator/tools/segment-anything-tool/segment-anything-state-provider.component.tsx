// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useEffect, useState } from 'react';

import { EncodingOutput } from '@geti/smart-tools/segment-anything';
import { toast } from '@geti/ui';
import { useQueryClient, UseQueryResult } from '@tanstack/react-query';
import { isEmpty } from 'lodash-es';

import { useAnnotator } from '../../annotator-provider.component';
import { Shape } from '../../types';
import { ModelLoading } from './model-loading.component';
import { InteractiveAnnotationPoint } from './segment-anything.interface';
import { useDecodingMutation, useDecodingQuery, useDecodingQueryOptions } from './use-decoding-query.hook';
import { useSegmentAnythingModel } from './use-segment-anything.hook';
import { useSingleStackFn } from './use-single-stack-fn.hook';

interface SegmentAnythingState {
    points: InteractiveAnnotationPoint[];
}

interface SegmentAnythingStateContextProps {
    isProcessing: boolean;
    isLoading: boolean;

    points: InteractiveAnnotationPoint[];
    addPoint: (point: InteractiveAnnotationPoint, keepPreviousPoints?: boolean) => void;
    result: { shapes: Shape[] };
    handleCancelAnnotation: () => void;
    handleConfirmAnnotation: () => void;

    encodingQuery: UseQueryResult<EncodingOutput>;
    decodingQueryFn: (points: InteractiveAnnotationPoint[]) => Promise<Shape[]>;
}

const SegmentAnythingStateContext = createContext<SegmentAnythingStateContextProps | undefined>(undefined);

export const SegmentAnythingStateProvider = ({ children }: { children: ReactNode }) => {
    const [state, setState] = useState<SegmentAnythingState>({
        points: [],
    });
    const [_, setIsDrawing] = useState(false);

    const { addAnnotation } = useAnnotator();

    const queryClient = useQueryClient();
    const { encodingQuery, decodingQueryFn, isLoading } = useSegmentAnythingModel();
    const throttledDecodingQueryFn = useSingleStackFn(decodingQueryFn);
    const decodingQueryOptions = useDecodingQueryOptions(state.points, throttledDecodingQueryFn);
    const decodingQuery = useDecodingQuery(state.points, throttledDecodingQueryFn);

    useEffect(() => {
        if (state.points.length > 0 && decodingQuery.data !== undefined && decodingQuery.data.length === 0) {
            if (!decodingQuery.isPlaceholderData) {
                toast({
                    message: `Unable to segment object from the selected point${
                        state.points.length > 1 ? 's' : ''
                    }. Press ESC to reset points.`,
                    type: 'warning',
                });
            }
        }
    }, [decodingQuery.data, decodingQuery.isPlaceholderData, state.points]);

    const decodingMutation = useDecodingMutation(decodingQueryFn);

    const reset = async () => {
        queryClient.removeQueries({ queryKey: decodingQueryOptions.queryKey });

        setIsDrawing(false);
    };

    useEffect(() => {
        return () => setIsDrawing(false);
    }, [setIsDrawing]);

    const isProcessing = decodingQuery.isFetching;

    const hasResults = decodingQuery.data && !isEmpty(decodingQuery.data) && !isEmpty(state.points);
    const outputShapes = decodingQuery.data ?? [];
    const handleConfirmAnnotation = () => {
        if (isProcessing) {
            return;
        }

        if (hasResults) {
            outputShapes.map((shape) => addAnnotation(shape));
        }

        reset();
    };

    const handleCancelAnnotation = () => {
        if (!isProcessing) {
            reset();
        }
    };

    const addPoint = (point: InteractiveAnnotationPoint, keepPreviousData = false) => {
        setIsDrawing(true);

        if (keepPreviousData) {
            setState((r) => ({ points: [...r.points, point] }));
        } else {
            decodingMutation.mutateAsync([point]).then(() => {
                setIsDrawing(false);
            });
        }
    };

    if (isLoading || encodingQuery.data === undefined) {
        return <ModelLoading isLoadingModel={isLoading} />;
    }

    return (
        <SegmentAnythingStateContext.Provider
            value={{
                isProcessing,
                result: { shapes: outputShapes },
                isLoading,
                points: state.points,
                addPoint,
                handleConfirmAnnotation,
                handleCancelAnnotation,
                encodingQuery,
                decodingQueryFn,
            }}
        >
            {children}
        </SegmentAnythingStateContext.Provider>
    );
};

export const useSegmentAnything = (): SegmentAnythingStateContextProps => {
    const context = useContext(SegmentAnythingStateContext);

    if (context === undefined) {
        throw new Error('useSegmentAnythingState must be used within a SegmentAnythingStateProvider');
    }

    return context;
};
