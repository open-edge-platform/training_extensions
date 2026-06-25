// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState, type RefObject } from 'react';

import { useResizeObserver } from '@react-aria/utils';

export const useContainerSize = (ref: RefObject<HTMLElement | null>) => {
    const [size, setSize] = useState({ width: 100, height: 100 });
    useResizeObserver({
        ref,
        box: 'border-box',
        onResize: () => {
            if (!ref.current) {
                return;
            }

            if (size.width === ref.current.clientWidth && size.height === ref.current.clientHeight) {
                return;
            }

            setSize({
                width: ref.current.clientWidth,
                height: ref.current.clientHeight,
            });
        },
    });

    return size;
};
