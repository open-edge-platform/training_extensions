// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Circle as CircleInterface } from '../interfaces';

interface CircleProps {
    circle: CircleInterface;
    styles: React.SVGProps<SVGCircleElement>;
    ariaLabel: string;
}
export const Circle = ({ circle, styles, ariaLabel, ...rest }: CircleProps) => {
    return <circle cx={circle.x} cy={circle.y} r={circle.r} {...styles} aria-label={ariaLabel} {...rest} />;
};
