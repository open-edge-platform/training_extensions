// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, renderHook } from '@testing-library/react';

import { KeyMap } from '../../constants/keyboard.interface';
import { usePanning } from './use-pannig.hook';

describe('usePanning', () => {
    it('returns false by default', () => {
        const { result } = renderHook(() => usePanning());
        expect(result.current).toBe(false);
    });

    it('sets isPanning to true on Space keydown', () => {
        const { result } = renderHook(() => usePanning());

        fireEvent.keyDown(document.body, { code: KeyMap.Space });
        expect(result.current).toBe(true);
    });

    it('sets isPanning to false on Space keyup after keydown', () => {
        const { result } = renderHook(() => usePanning());

        fireEvent.keyDown(document.body, { code: KeyMap.Space });
        expect(result.current).toBe(true);

        fireEvent.keyUp(document.body, { code: KeyMap.Space });
        expect(result.current).toBe(false);
    });

    it('does not set isPanning to true for other keys', () => {
        const { result } = renderHook(() => usePanning());

        fireEvent.keyDown(document.body, { code: KeyMap.Enter });
        expect(result.current).toBe(false);
    });
});
