// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, renderHook } from '@testing-library/react';

import { useDebouncedCallback } from './use-debounced-callback.hook';

describe('useDebouncedCallback', () => {
    it('executes a given callback after given delay', () => {
        vi.useFakeTimers();

        const mockCallback = vi.fn();
        const delay = 1000;
        const { result } = renderHook(() => useDebouncedCallback(mockCallback, delay));

        const debouncedCallback = result.current;

        act(() => {
            debouncedCallback();
        });

        expect(mockCallback).toHaveBeenCalledTimes(0);

        vi.advanceTimersByTime(delay);

        expect(mockCallback).toHaveBeenCalledTimes(1);

        vi.useRealTimers();
    });
});
