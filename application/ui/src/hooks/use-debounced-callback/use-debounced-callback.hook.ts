// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useLayoutEffect, useMemo, useRef } from 'react';

import { debounce, type DebouncedFunc } from 'lodash-es';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Callback = (...args: any[]) => void;

export const useDebouncedCallback = (callback: Callback, delay: number): DebouncedFunc<Callback> => {
    const savedCallback = useRef(callback);

    useLayoutEffect(() => {
        savedCallback.current = callback;
    }, [callback]);

    const debouncedCallback = useMemo(() => debounce(savedCallback.current, delay), [delay]);

    useEffect(() => {
        return () => {
            debouncedCallback.cancel?.();
        };
    }, []);

    return debouncedCallback;
};
