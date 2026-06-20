// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState, useEffect, useCallback, type RefObject } from 'react';

import { useResizeObserver } from '@react-aria/utils';

export const useContainerSize = (ref: RefObject<HTMLElement | null>) => {
    const [size, setSize] = useState({ width: 100, height: 100 });

    const updateSize = useCallback(() => {
        if (!ref.current) {
            return;
        }

        setSize((prevSize) => {
            if (prevSize.width === ref.current!.clientWidth && prevSize.height === ref.current!.clientHeight) {
                return prevSize;
            }

            return {
                width: ref.current!.clientWidth,
                height: ref.current!.clientHeight,
            };
        });
    }, [ref]);

    useEffect(() => {
        updateSize();
        window.addEventListener('resize', updateSize);
        return () => window.removeEventListener('resize', updateSize);
    }, [updateSize]);

    useResizeObserver({
        ref,
        box: 'border-box',
        onResize: updateSize,
    });

    return size;
};
