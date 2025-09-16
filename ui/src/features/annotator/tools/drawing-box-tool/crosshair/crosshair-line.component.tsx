// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Point } from '../../../types';

export enum CROSSHAIR_LINE_DIRECTION {
    HORIZONTAL,
    VERTICAL,
}

const DEFAULT_SIZE = 1.0;

interface CrosshairLineProps {
    zoom: number;
    point: Point;
    direction: CROSSHAIR_LINE_DIRECTION;
}

const colors = {
    main: {
        color: 'white',
        opacity: 0.9,
    },
    shade: {
        color: '#000000',
        opacity: 0.12,
    },
};

export const CrosshairLine = ({ zoom, direction, point }: CrosshairLineProps) => {
    const attributes =
        direction === CROSSHAIR_LINE_DIRECTION.HORIZONTAL
            ? {
                  y: point.y,
                  width: '100%',
                  height: DEFAULT_SIZE / zoom,
              }
            : {
                  x: point.x,
                  width: DEFAULT_SIZE / zoom,
                  height: '100%',
              };

    return (
        <rect
            {...attributes}
            fillOpacity={colors.main.opacity}
            fill={colors.main.color}
            stroke={colors.shade.color}
            strokeOpacity={colors.shade.opacity}
            strokeWidth={DEFAULT_SIZE / zoom}
        />
    );
};
