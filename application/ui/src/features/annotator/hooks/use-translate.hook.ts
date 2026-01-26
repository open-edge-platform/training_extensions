// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useRef } from 'react';

import { isLeftButton } from '../../../shared/buttons-utils';
import type { Point } from '../../../shared/types';

type UseTranslateProps = {
    zoom: number;
    onTranslate: (delta: Point) => void;
    onComplete: () => void;
};
export const useTranslate = ({ zoom, onTranslate, onComplete }: UseTranslateProps) => {
    const dragFromPoint = useRef<null | Point>(null);

    const onPointerDown = (event: PointerEvent<SVGSVGElement>): void => {
        if (dragFromPoint.current !== null) {
            return;
        }

        if (event.pointerType === 'touch' || !isLeftButton(event)) {
            return;
        }

        const mouse = { x: Math.round(event.clientX / zoom), y: Math.round(event.clientY / zoom) };

        event.currentTarget.setPointerCapture(event.pointerId);

        dragFromPoint.current = mouse;
    };

    const onPointerMove = (event: PointerEvent<SVGSVGElement>) => {
        event.preventDefault();

        if (dragFromPoint.current === null) {
            return;
        }

        const mouse = { x: Math.round(event.clientX / zoom), y: Math.round(event.clientY / zoom) };

        onTranslate({
            x: mouse.x - dragFromPoint.current.x,
            y: mouse.y - dragFromPoint.current.y,
        });

        dragFromPoint.current = mouse;
    };

    const onPointerUp = (event: PointerEvent<SVGSVGElement>) => {
        if (dragFromPoint.current === null) {
            return;
        }

        event.preventDefault();
        dragFromPoint.current = null;
        event.currentTarget.releasePointerCapture(event.pointerId);
        onComplete();
    };

    return { onPointerDown, onPointerMove, onPointerUp };
};
