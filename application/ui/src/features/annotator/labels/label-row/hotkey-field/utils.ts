// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isMac } from '@react-aria/utils';

const CTRL_KEY = 'ctrl';
const CTRL_KEY_DESCRIPTION = 'control';
const COMMAND_KEY = 'meta';
const COMMAND_KEY_DESCRIPTION = 'cmd';

export enum KeyboardEvents {
    KeyDown = 'keydown',
    KeyUp = 'keyup',
}

export enum KeyMap {
    Z = 'Z',
    z = 'z',
    Space = ' ',
    Esc = 'Escape',
    Enter = 'Enter',
    Delete = 'Delete',
    ArrowUp = 'ArrowUp',
    ArrowDown = 'ArrowDown',
    Backspace = 'Backspace',

    Alt = 'Alt',
    Meta = 'Meta',
    Command = 'Cmd',
    Shift = 'Shift',
    Control = 'Ctrl',
}

export const CTRL_OR_COMMAND_KEY = isMac() ? COMMAND_KEY : CTRL_KEY;

export const getKeyName = (key: string): string => {
    const lowerCaseKey = key.toLowerCase();

    if (lowerCaseKey.includes(CTRL_KEY_DESCRIPTION)) {
        return lowerCaseKey.replace(CTRL_KEY_DESCRIPTION, CTRL_KEY).toLocaleUpperCase();
    }

    if (lowerCaseKey.includes(COMMAND_KEY)) {
        return lowerCaseKey.replace(COMMAND_KEY, COMMAND_KEY_DESCRIPTION).toLocaleUpperCase();
    }

    return key.toLocaleUpperCase();
};

export const isModifierKey = (key = ''): boolean => {
    return [
        KeyMap.Alt.toLocaleUpperCase(),
        KeyMap.Meta.toLocaleUpperCase(),
        KeyMap.Command.toLocaleUpperCase(),
        KeyMap.Shift.toLocaleUpperCase(),
        KeyMap.Control.toLocaleUpperCase(),
    ].includes(key.toLocaleUpperCase());
};

export const getKeyBoardKey = (event: KeyboardEvent) => {
    const { key, code, shiftKey } = event;

    if (shiftKey && code.includes('Digit')) {
        return code[code.length - 1];
    }

    return getKeyName(key);
};
