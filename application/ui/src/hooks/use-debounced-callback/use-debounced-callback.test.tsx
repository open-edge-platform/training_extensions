// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { act, renderHook } from '@testing-library/react';

import { useDebouncedCallback } from './use-debounced-callback.hook';

describe('useDebouncedCallback', () => {
    it('executes a given callback after given delay', () => {
        jest.useFakeTimers();

        const mockCallback = jest.fn();
        const delay = 1000;
        const { result } = renderHook(() => useDebouncedCallback(mockCallback, delay));

        const debouncedCallback = result.current;

        act(() => {
            debouncedCallback();
        });

        expect(mockCallback).toHaveBeenCalledTimes(0);

        jest.advanceTimersByTime(delay);

        expect(mockCallback).toHaveBeenCalledTimes(1);

        jest.clearAllTimers();
    });
});
