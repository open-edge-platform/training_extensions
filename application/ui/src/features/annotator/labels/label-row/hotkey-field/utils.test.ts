// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getKeyBoardKey, isModifierKey } from './utils';

describe('hotkey-edition-field utils', () => {
    it('isModifierKey', () => {
        expect(isModifierKey('Alt')).toBe(true);
        expect(isModifierKey('Meta')).toBe(true);
        expect(isModifierKey('Shift')).toBe(true);
        expect(isModifierKey('Ctrl')).toBe(true);
        expect(isModifierKey('1')).toBe(false);
        expect(isModifierKey('A')).toBe(false);
        expect(isModifierKey('ArrowUp')).toBe(false);
    });

    const mockKeyboardEvent = (key: string, code: string, shiftKey = false): KeyboardEvent =>
        ({ key, code, shiftKey }) as unknown as KeyboardEvent;

    it('getKeyBoardKey', () => {
        expect(getKeyBoardKey(mockKeyboardEvent('ArrowUp', 'ArrowUp'))).toBe('ARROWUP');
        expect(getKeyBoardKey(mockKeyboardEvent('d', 'KeyD'))).toBe('D');
        expect(getKeyBoardKey(mockKeyboardEvent('1', 'Digit1', true))).toBe('1');
        expect(getKeyBoardKey(mockKeyboardEvent('2', 'Digit2', true))).toBe('2');
        expect(getKeyBoardKey(mockKeyboardEvent('3', 'Digit3'))).toBe('3');
    });
});
