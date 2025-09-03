// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Rect as RectInterface } from './interfaces';

interface RectangleProps {
    rect: RectInterface;
    styles: React.SVGProps<SVGRectElement>;
    ariaLabel: string;
}
export const Rectangle = ({ rect, styles, ariaLabel }: RectangleProps) => {
    const { x, y, width, height } = rect;

    return <rect x={x} y={y} width={width} height={height} {...styles} aria-label={ariaLabel} />;
};
