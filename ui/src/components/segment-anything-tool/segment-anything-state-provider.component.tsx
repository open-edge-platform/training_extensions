// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, useContext, useEffect } from 'react';

import { EncodingOutput } from '@geti/smart-tools/segment-anything';
import { toast } from '@geti/ui';
import { useQueryClient, UseQueryResult } from '@tanstack/react-query';
import { isEmpty } from 'lodash-es';

import { Shape } from '../shapes/interfaces';
import { InteractiveAnnotationPoint } from './segment-anything.interface';
import { useDecodingMutation, useDecodingQuery, useDecodingQueryOptions } from './use-decoding-query.hook';
import { useSegmentAnythingModel } from './use-segment-anything-model.hook';
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

export const SegmentAnythingStateProvider = ({ children }: StateProviderProps) => {
    const [state, setState, undoRedoActions] = useUndoRedoState<SegmentAnythingState>({
        points: [],
    });

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

    const { addShapes, setIsDrawing } = useAnnotationScene();

    const reset = async () => {
        queryClient.removeQueries({ queryKey: decodingQueryOptions.queryKey });
        undoRedoActions.reset({ points: [] });

        setIsDrawing(false);
    };

    useAddUnfinishedShape({
        shapes: state.points.length > 0 && decodingQuery.data ? decodingQuery.data : [],
        addShapes: (unfinishedShapes) => {
            reset();

            if (unfinishedShapes.length === 0) {
                return [];
            }

            const shapes = addShapes(unfinishedShapes as Shape[]);
            return shapes;
        },
        reset,
    });
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
            addShapes(outputShapes);
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
            undoRedoActions.reset({ points: [] });
            decodingMutation.mutateAsync([point]).then(() => {
                setIsDrawing(false);
            });
        }
    };

    const annotationToolContext = useAnnotationToolContext();
    useApplyLabelToPendingAnnotations({
        applyAnnotations: handleConfirmAnnotation,
        annotationToolContext,
        tool: ToolType.SegmentAnythingTool,
    });

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
            <UndoRedoProvider state={undoRedoActions}>{children}</UndoRedoProvider>
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
