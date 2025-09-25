// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo } from 'react';

import { throttle, type DebouncedFunc } from 'lodash-es';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Callback = (...args: any[]) => void;

export const useThrottledCallback = (callback: Callback, delay: number): DebouncedFunc<Callback> => {
    const debouncedCallback = useMemo(() => {
        return throttle(callback, delay, {
            leading: true,
            trailing: true,
        });
    }, [callback, delay]);

    useEffect(() => {
        return () => {
            debouncedCallback.cancel();
        };
    }, [debouncedCallback]);

    return debouncedCallback;
};
