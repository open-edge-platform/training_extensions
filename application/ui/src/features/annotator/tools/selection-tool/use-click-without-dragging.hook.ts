// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { RefObject, useState } from 'react';

import { useEventListener } from 'hooks/event-listener.hook';

export const useClickWithoutDragging = (
    ref: RefObject<SVGSVGElement | null>,
    onClick: (event: globalThis.PointerEvent) => void
) => {
    const [isDragging, setIsDragging] = useState(false);
    const handleClick = (event: globalThis.PointerEvent): void => {
        event.preventDefault();

        if (isDragging) {
            return;
        }

        onClick(event);
    };

    useEventListener('pointerup', handleClick, ref);
    useEventListener('pointerdown', () => setIsDragging(false), ref);
    useEventListener('pointermove', () => setIsDragging(true), ref);
};
