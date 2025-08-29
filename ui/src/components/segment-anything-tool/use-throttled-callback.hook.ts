// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useLayoutEffect, useMemo, useRef } from 'react';

import { throttle, type DebouncedFunc } from 'lodash-es';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Callback = (...args: any[]) => void;

export const useThrottledCallback = (callback: Callback, delay: number): DebouncedFunc<Callback> => {
    const savedCallback = useRef(callback);

    useLayoutEffect(() => {
        savedCallback.current = callback;
    }, [callback]);

    const debouncedCallback = useMemo(() => {
        return throttle(savedCallback.current, delay, {
            leading: true,
            trailing: true,
        });
    }, [delay]);

    useEffect(() => {
        return () => {
            debouncedCallback.cancel();
        };
    }, [debouncedCallback]);

    return debouncedCallback;
};
