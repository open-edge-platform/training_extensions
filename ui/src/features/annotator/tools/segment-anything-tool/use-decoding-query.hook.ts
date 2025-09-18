// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { queryOptions, useMutation, useQuery } from '@tanstack/react-query';

import { useAnnotator } from '../../annotator-provider.component';
import { RegionOfInterest, Shape } from '../../types';
import { removeOffLimitPoints } from '../utils';
import { InteractiveAnnotationPoint } from './segment-anything.interface';

const roi: RegionOfInterest = {
    x: 0,
    y: 0,
    width: 100,
    height: 100,
};

export const useDecodingQueryOptions = (
    points: InteractiveAnnotationPoint[],
    queryFn: (points: InteractiveAnnotationPoint[]) => Promise<Shape[]>
) => {
    const { mediaItem } = useAnnotator();
    // Round points so that when the user slightly moves their mouse we do not
    // immediately recompute the decoding
    const roundedPoints = points.map((point) => ({
        x: Math.round(point.x),
        y: Math.round(point.y),
        positive: point.positive,
    }));

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
    return useMutation({
        mutationFn: async (points: InteractiveAnnotationPoint[]) => {
            // Round points so that when the user slightly moves their mouse we do not
            // immediately recompute the decoding
            const roundedPoints = points.map((point) => ({
                x: Math.round(point.x),
                y: Math.round(point.y),
                positive: point.positive,
            }));

            // TODO: Add callback to add shapes
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const shapes = (await queryFn(roundedPoints)).map((shape) => {
                return removeOffLimitPoints(shape, roi);
            });

            // Add the shapes to the canvas here
            // addShapes(shapes);
        },
    });
};
