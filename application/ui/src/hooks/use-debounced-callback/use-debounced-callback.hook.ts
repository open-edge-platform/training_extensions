// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

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
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return debouncedCallback;
};
