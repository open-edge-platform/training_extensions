// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getKeyName } from '../utils';
import { Action, HOTKEY_EDITION_ACTION } from './actions';

export interface HotkeyFieldState {
    isFocused: boolean;
    isDirty: boolean;
    keys: string;
}

export const reducer = (state: HotkeyFieldState, action: Action): HotkeyFieldState => {
    const MAX_KEYS_IN_HOTKEY = 2;

    switch (action.type) {
        case HOTKEY_EDITION_ACTION.CHANGE_FOCUS:
            return {
                isDirty: false,
                isFocused: action.isFocused,
                keys: state.isDirty ? state.keys : '',
            };

        case HOTKEY_EDITION_ACTION.ADD_KEY:
            const keys = action.keys.slice(-MAX_KEYS_IN_HOTKEY).map((key: string) => getKeyName(key));

            if (!keys.length) {
                return state;
            }

            return {
                ...state,
                isDirty: true,
                keys: keys.join('+'),
            };

        default:
            return state;
    }
};
