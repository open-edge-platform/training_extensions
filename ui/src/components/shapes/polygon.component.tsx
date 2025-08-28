// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { Point, Polygon as PolygonInterface } from './interfaces';

const getFormattedPoints = (points: Point[]): string => points.map(({ x, y }) => `${x},${y}`).join(' ');

interface PolygonProps {
    polygon: PolygonInterface;
    styles: React.SVGProps<SVGPolygonElement>;
    ariaLabel: string;
}

export const Polygon = ({ polygon, styles, ariaLabel }: PolygonProps) => {
    const points = useMemo((): string => getFormattedPoints(polygon.points), [polygon]);

    return (
        <g>
            <polygon points={points} {...styles} aria-label={ariaLabel} />
        </g>
    );
};
