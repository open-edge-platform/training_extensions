// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { queryOptions, useMutation, useQuery } from '@tanstack/react-query';

import { useAnnotator } from '../../annotator-provider.component';
import { Shape } from '../../types';
import { removeOffLimitPoints } from '../utils';
import { InteractiveAnnotationPoint } from './segment-anything.interface';

const roundPoint = (point: InteractiveAnnotationPoint): InteractiveAnnotationPoint => ({
    x: Math.round(point.x),
    y: Math.round(point.y),
    positive: point.positive,
});

export const useDecodingQueryOptions = (
    points: InteractiveAnnotationPoint[],
    queryFn: (points: InteractiveAnnotationPoint[]) => Promise<Shape[]>
) => {
    const { mediaItem, roi } = useAnnotator();
    // Round points so that when the user slightly moves their mouse we do not
    // immediately recompute the decoding
    const roundedPoints = points.map(roundPoint);

    return queryOptions({
        queryKey: ['segment-anything-model', 'decoding', mediaItem?.id, roundedPoints, roi],
        queryFn: async () => {
            const shapes = await queryFn(roundedPoints);

            return shapes.map((shape) => {
                return removeOffLimitPoints(shape, roi);
            });
        },
        staleTime: Infinity,
        retry: 0,
    });
};

export const useDecodingQuery = (
    points: InteractiveAnnotationPoint[],
    queryFn: (points: InteractiveAnnotationPoint[]) => Promise<Shape[]>
) => {
    const decodingQueryOptions = useDecodingQueryOptions(points, queryFn);

    return useQuery(decodingQueryOptions);
};

export const useDecodingMutation = (queryFn: (points: InteractiveAnnotationPoint[]) => Promise<Shape[]>) => {
    const { addAnnotation, roi } = useAnnotator();

    return useMutation({
        mutationFn: async (points: InteractiveAnnotationPoint[]) => {
            // Round points so that when the user slightly moves their mouse we do not
            // immediately recompute the decoding
            const roundedPoints = points.map(roundPoint);

            const shapes = (await queryFn(roundedPoints)).map((shape) => {
                return removeOffLimitPoints(shape, roi);
            });

            shapes.map((shape) => addAnnotation(shape));
        },
    });
};
