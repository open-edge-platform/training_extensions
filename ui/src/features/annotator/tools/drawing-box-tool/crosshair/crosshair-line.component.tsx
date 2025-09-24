// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Point } from '../../../types';

const DEFAULT_SIZE = 1.0;

interface CrosshairLineProps {
    zoom: number;
    point: Point;
    direction: 'horizontal' | 'vertical';
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

export const CrosshairLine = ({ direction, point }: CrosshairLineProps) => {
    const sizeRatio = `calc(${DEFAULT_SIZE} / var(--zoom-scale))`;
    const attributes =
        direction === 'horizontal'
            ? {
                  y: point.y,
                  width: '100%',
                  height: sizeRatio,
              }
            : {
                  x: point.x,
                  width: sizeRatio,
                  height: '100%',
              };

    return (
        <rect
            {...attributes}
            fillOpacity={colors.main.opacity}
            fill={colors.main.color}
            stroke={colors.shade.color}
            strokeOpacity={colors.shade.opacity}
            strokeWidth={sizeRatio}
        />
    );
};
