// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { RefObject, useLayoutEffect, useState } from 'react';

import { useResizeObserver } from '@react-aria/utils';

export function useSizeHook<T extends HTMLElement>(target: RefObject<T | null>): DOMRect | undefined {
    const [size, setSize] = useState<DOMRect>();

    useLayoutEffect(() => {
        if (target.current !== null) {
            setSize(target.current.getBoundingClientRect());
        }
    }, [target]);

    useResizeObserver({
        ref: target,
        box: 'border-box',
        onResize: () => setSize(target.current?.getBoundingClientRect()),
    });

    return size;
}
