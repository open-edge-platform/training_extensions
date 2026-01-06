// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { Polygon } from '../../types';
import { getFormattedPoints, ShapeStyle } from './utils';

const CIRCLE_STROKE_COLOR = 'var(--energy-blue-shade)';

export interface PolygonDrawProps extends ShapeStyle<SVGPolygonElement> {
    shape: Polygon;
    ariaLabel?: string;
    indicatorRadius?: number;
}

export const PolygonDraw = ({ shape, styles, indicatorRadius, className = '', ariaLabel = '' }: PolygonDrawProps) => {
    return (
        <g>
            <circle
                r={indicatorRadius}
                cx={shape.points[0].x}
                cy={shape.points[0].y}
                fill='transparent'
                stroke={CIRCLE_STROKE_COLOR}
            />
            <polyline
                {...styles}
                points={getFormattedPoints(shape.points)}
                className={className}
                aria-label={ariaLabel}
            />
            ;
        </g>
    );
};
