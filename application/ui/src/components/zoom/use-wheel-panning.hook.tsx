// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useRef, useState } from 'react';

import { isEmpty } from 'lodash-es';

import { isWheelButton } from '../../features/annotator/buttons-utils';
import type { Point } from './types';

export const useWheelPanning = (setIsPanning: (value: boolean) => void) => {
    const [isGrabbing, setIsGrabbing] = useState(false);
    const lastPos = useRef<Point | null>(null);

    return {
        isGrabbing,
        onMouseLeave: () => {
            setIsPanning(false);
            setIsGrabbing(false);
            lastPos.current = null;
        },
        onPointerMove: (callback: (data: Point) => void) => (event: PointerEvent<HTMLDivElement>) => {
            if (isWheelButton(event) && !isEmpty(lastPos.current)) {
                const dx = event.clientX - lastPos.current.x;
                const dy = event.clientY - lastPos.current.y;
                lastPos.current = { x: event.clientX, y: event.clientY };

                callback({ x: dx, y: dy });
            }
        },
        onPointerDown: (event: PointerEvent<HTMLDivElement>) => {
            setIsGrabbing(true);

            if (isWheelButton(event)) {
                setIsPanning(true);
                lastPos.current = { x: event.clientX, y: event.clientY };
            }
        },
        onPointerUp: () => {
            setIsPanning(false);
            setIsGrabbing(false);
            lastPos.current = null;
        },
    };
};
