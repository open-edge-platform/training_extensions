// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, renderHook } from '@testing-library/react';

import { usePanning } from './use-panning.hook';

describe('usePanning', () => {
    it('returns false by default', () => {
        const { result } = renderHook(() => usePanning());
        expect(result.current).toBe(false);
    });

    it('sets isPanning to true on Space keydown', () => {
        const { result } = renderHook(() => usePanning());

        fireEvent.keyDown(document.body, { code: 'Space' });
        expect(result.current).toBe(true);
    });

    it('sets isPanning to false on Space keyup after keydown', () => {
        const { result } = renderHook(() => usePanning());

        fireEvent.keyDown(document.body, { code: 'Space' });
        expect(result.current).toBe(true);

        fireEvent.keyUp(document.body, { code: 'Space' });
        expect(result.current).toBe(false);
    });

    it('does not set isPanning to true for other keys', () => {
        const { result } = renderHook(() => usePanning());

        fireEvent.keyDown(document.body, { code: 'Enter' });
        expect(result.current).toBe(false);
    });
});
