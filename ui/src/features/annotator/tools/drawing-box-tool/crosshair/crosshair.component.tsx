// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Point } from '../../../types';
import { CrosshairLine } from './crosshair-line.component';

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
            <CrosshairLine zoom={zoom} point={location} direction={'horizontal'} />
            <CrosshairLine zoom={zoom} point={location} direction={'vertical'} />
        </g>
    );
};
