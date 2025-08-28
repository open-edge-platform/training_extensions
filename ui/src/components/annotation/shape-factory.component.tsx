// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Shape } from '../shapes/interfaces';
import { Polygon } from '../shapes/polygon.component';
import { Rectangle } from '../shapes/rectangle.component';

interface ShapeFactoryProps {
    shape: Shape;
    styles: React.SVGProps<SVGPolygonElement & SVGRectElement>;
    ariaLabel: string;
}
export const ShapeFactory = ({ shape, styles, ariaLabel }: ShapeFactoryProps) => {
    if (shape.shapeType === 'rect') {
        return <Rectangle rect={shape} styles={styles} ariaLabel={ariaLabel} />;
    } else {
        return <Polygon polygon={shape} styles={styles} ariaLabel={ariaLabel} />;
    }
};
