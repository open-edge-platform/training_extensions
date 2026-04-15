// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, type DependencyList } from 'react';

import { debounce } from 'lodash-es';

type AnyFunction = (...args: never[]) => void;

export const useDebounce = <T extends AnyFunction>(callback: T, delay: number, deps: DependencyList) => {
    const debouncedCallback = useMemo(() => debounce(callback, delay), deps);

    useEffect(() => {
        return () => {
            debouncedCallback.cancel();
        };
    }, [debouncedCallback]);

    return debouncedCallback;
};
