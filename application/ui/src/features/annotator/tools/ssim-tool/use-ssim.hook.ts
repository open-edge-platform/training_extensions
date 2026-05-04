// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useMemo, useState } from 'react';

import type { RunSSIMProps as ToolRunSSIMProps, SSIMMatch as ToolSSIMMatch } from '@geti/smart-tools';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Remote, wrap } from 'comlink';

import type { Rect, RegionOfInterest, Shape } from '../../../../shared/types';
import type { SSIMWorkerApi, SSIMWorkerInstance } from '../../webworkers/ssim-worker.interface';
import {
    convertToolShapeToGetiShape,
    getBoundingRectFromShape,
    intersectionOverUnion,
    isRectWithinRoi,
} from '../utils';

const MAX_NUMBER_ITEMS = 500;

type SSIMMatch = Omit<ToolSSIMMatch, 'shape'> & {
    shape: Rect;
};

type SSIMShapeType = Extract<Shape['type'], 'rectangle' | 'polygon'>;

type RunSSIMProps = Omit<ToolRunSSIMProps, 'template' | 'existingAnnotations' | 'shapeType'> & {
    template: Rect;
    existingAnnotations: Shape[];
    shapeType: SSIMShapeType;
};

type SSIMState = {
    shapes: Shape[];
    matches: SSIMMatch[];
    threshold: number;
};

const INITIAL_SSIM_STATE: SSIMState = {
    shapes: [],
    matches: [],
    threshold: 0,
};

export const useSSIMWorker = (enabled = true) => {
    const { data, isLoading, isError } = useQuery<{ worker: Worker; instance: Remote<SSIMWorkerInstance> }>({
        queryKey: ['workers', 'SSIM'],
        queryFn: async ({ signal }) => {
            const worker = new Worker(new URL('../../webworkers/ssim-worker', import.meta.url), {
                type: 'module',
            });
            // Terminate the worker if the query is cancelled (e.g. annotator unmounts)
            // before build() resolves, so we don't leak the in-flight worker.
            signal.addEventListener('abort', worker.terminate, { once: true });

            try {
                const instance = await wrap<SSIMWorkerApi>(worker).build();

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

    return { worker: data?.instance, isLoading, isError };
};

const toToolRect = (rect: Rect): ToolRunSSIMProps['template'] => {
    const { x, y, width, height } = rect;

    return {
        x,
        y,
        width,
        height,
        shapeType: 'rect',
    };
};

const toToolRunSSIMProps = ({
    imageData,
    roi,
    template,
    existingAnnotations,
    autoMergeDuplicates,
}: RunSSIMProps): ToolRunSSIMProps => {
    return {
        imageData,
        roi,
        template: toToolRect(template),
        existingAnnotations: existingAnnotations
            .map(getBoundingRectFromShape)
            .filter((shape): shape is Rect => shape !== null)
            .map(toToolRect),
        autoMergeDuplicates,
        shapeType: 'rect',
    };
};

const convertToolMatchesToGetiMatches = (matches: ToolSSIMMatch[]): SSIMMatch[] => {
    return matches.map((match) => ({
        ...match,
        shape: convertToolShapeToGetiShape(match.shape),
    }));
};

const filterSSIMResults = (
    roi: RegionOfInterest,
    items: SSIMMatch[],
    template: Rect,
    filter: Rect[],
    maxItems = MAX_NUMBER_ITEMS,
    overlapThreshold = 0.2
): SSIMMatch[] => {
    const collector: SSIMMatch[] = [{ shape: template, confidence: 1 }];
    const filterAsMatches = filter.map((shape) => ({ shape, confidence: 1 }));
    const filteredItems = items.filter(({ shape }) => isRectWithinRoi(roi, shape));

    for (const value of filteredItems) {
        const overlapsWithExisting = [...filterAsMatches, ...collector].some(
            (otherMatch) => intersectionOverUnion(otherMatch.shape, value.shape) > overlapThreshold
        );

        if (!overlapsWithExisting) {
            collector.push(value);
        }

        if (collector.length === maxItems) {
            return collector;
        }
    }

    return collector;
};

export const guessNumberOfItemsThreshold = (matches: SSIMMatch[], confidenceThreshold = 0.9): number => {
    const guess = matches.findIndex(({ confidence }) => confidence < confidenceThreshold);

    return guess === -1 ? matches.length : guess;
};

const convertRectToShape = (rectangle: Rect, shapeType: SSIMShapeType): Shape => {
    if (shapeType === 'rectangle') {
        return rectangle;
    }

    return {
        type: 'polygon',
        points: [
            { x: rectangle.x, y: rectangle.y },
            { x: rectangle.x + rectangle.width, y: rectangle.y },
            { x: rectangle.x + rectangle.width, y: rectangle.y + rectangle.height },
            { x: rectangle.x, y: rectangle.y + rectangle.height },
        ],
    };
};

export const useSSIM = (enabled = true) => {
    const { worker: ssim, isLoading: isLoadingWorker, isError: isWorkerError } = useSSIMWorker(enabled);

    const [toolState, setToolState] = useState<SSIMState>(INITIAL_SSIM_STATE);
    const [previewThreshold, setPreviewThreshold] = useState<number | null>(null);
    const [ssimProps, setSSIMProps] = useState<RunSSIMProps | null>(null);

    const updateToolState = useCallback(
        (updatedProperties: Partial<SSIMState>, shapeType: SSIMShapeType = 'rectangle') => {
            if (updatedProperties.threshold !== undefined) {
                setPreviewThreshold(null);
            }

            setToolState((currentState) => {
                const newState = { ...currentState, ...updatedProperties };
                const matches = newState.matches.slice(0, newState.threshold);
                const shapes = matches.map(({ shape }) => convertRectToShape(shape, shapeType));

                return {
                    ...newState,
                    shapes,
                };
            });
        },
        []
    );

    const {
        mutate,
        reset: resetMutation,
        isPending,
        error,
    } = useMutation({
        mutationFn: async (runSSIMProps: RunSSIMProps) => {
            if (ssim === undefined) {
                throw new Error('SSIM worker is not initialized yet');
            }

            return ssim.executeSSIM(toToolRunSSIMProps(runSSIMProps));
        },
        onSuccess: (matches, { existingAnnotations, autoMergeDuplicates, template, roi, shapeType = 'rectangle' }) => {
            const ssimMatches = convertToolMatchesToGetiMatches(matches);
            const existingRects: Rect[] = autoMergeDuplicates
                ? existingAnnotations.map(getBoundingRectFromShape).filter((shape): shape is Rect => shape !== null)
                : [];
            const filteredMatches = filterSSIMResults(roi, ssimMatches, template, existingRects);
            const threshold = guessNumberOfItemsThreshold(filteredMatches);

            updateToolState(
                {
                    matches: filteredMatches,
                    threshold,
                },
                shapeType
            );
        },
    });

    const runSSIM = useCallback(
        (props: RunSSIMProps) => {
            setSSIMProps(props);
            mutate(props);
        },
        [mutate]
    );

    const rerun = useCallback(
        (props: Partial<RunSSIMProps>) => {
            if (ssimProps !== null) {
                runSSIM({ ...ssimProps, ...props });
            }
        },
        [ssimProps, runSSIM]
    );

    const reset = useCallback(() => {
        setToolState(INITIAL_SSIM_STATE);
        setSSIMProps(null);
        setPreviewThreshold(null);
        resetMutation();
    }, [resetMutation]);

    return useMemo(
        () => ({
            runSSIM,
            rerun,
            reset,
            updateToolState,
            toolState,
            previewThreshold,
            setPreviewThreshold,
            isLoading: isLoadingWorker,
            isProcessing: isPending,
            isError: isWorkerError || error !== null,
            worker: ssim as Remote<SSIMWorkerInstance> | undefined,
        }),
        [
            runSSIM,
            rerun,
            reset,
            updateToolState,
            toolState,
            previewThreshold,
            isLoadingWorker,
            isPending,
            isWorkerError,
            error,
            ssim,
        ]
    );
};
