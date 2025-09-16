// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Point } from '../../../types';
import { CROSSHAIR_LINE_DIRECTION, CrosshairLine } from './crosshair-line.component';

interface CrosshairProps {
    location: Point | null;
    zoom: number;
}

export const Crosshair = ({ location, zoom }: CrosshairProps) => {
    if (location === null) {
        return <g></g>;
    }

    return (
        <g>
            <CrosshairLine zoom={zoom} point={location} direction={CROSSHAIR_LINE_DIRECTION.HORIZONTAL} />
            <CrosshairLine zoom={zoom} point={location} direction={CROSSHAIR_LINE_DIRECTION.VERTICAL} />
        </g>
    );
};
