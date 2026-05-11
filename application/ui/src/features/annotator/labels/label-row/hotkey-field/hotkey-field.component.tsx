// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useReducer, useRef } from 'react';

import { TextField, TextFieldRef, type SpectrumTextFieldProps } from '@geti/ui';
import { useEventListener } from 'hooks/event-listener.hook';

import { HOTKEY_EDITION_ACTION } from './reducer/actions';
import { HotkeyFieldState, reducer } from './reducer/reducer';
import { getKeyBoardKey, isModifierKey, KeyboardEvents } from './utils';

interface HotkeyFieldProps {
    value: string;
    hasAutoFocus?: boolean;
    onEnter?: () => void;
    onChange: (value: string) => void;
}

const initialState: HotkeyFieldState = {
    isFocused: false,
    isDirty: true,
    keys: '',
};

const isEnter = (event: KeyboardEvent) => {
    return event.key === 'Enter';
};

export const HotkeyField = ({
    value,
    hasAutoFocus = false,
    onEnter,
    onChange,
    ...props
}: HotkeyFieldProps & SpectrumTextFieldProps) => {
    const [state, dispatch] = useReducer(reducer, initialState);
    const textFieldRef = useRef<TextFieldRef>(null);
    const hotKeysPressed = useRef<string[]>([]);

    useEffect(() => {
        if (hasAutoFocus) {
            textFieldRef.current?.focus();
        }
    }, [hasAutoFocus]);

    useEffect(() => {
        state.keys && onChange(state.keys);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [state.keys]);

    const onFocusChange = (isFocused: boolean) => {
        dispatch({ type: HOTKEY_EDITION_ACTION.CHANGE_FOCUS, isFocused });

        props.onFocusChange && props.onFocusChange(isFocused);
    };

    useEventListener(KeyboardEvents.KeyDown, (event: KeyboardEvent) => {
        const { repeat } = event;
        const newKey = getKeyBoardKey(event);

        const [currentKey] = hotKeysPressed.current;
        const hasMultipleModifiers = isModifierKey(currentKey) && isModifierKey(newKey);

        if (isEnter(event)) {
            return;
        }

        if (state.isFocused && !repeat && !hasMultipleModifiers) {
            hotKeysPressed.current.push(newKey);
            event.preventDefault();
        }
    });

    useEventListener(KeyboardEvents.KeyUp, (event: KeyboardEvent) => {
        if (state.isFocused && isEnter(event)) {
            onEnter?.();
            return;
        }

        if (state.isFocused) {
            dispatch({ type: HOTKEY_EDITION_ACTION.ADD_KEY, keys: hotKeysPressed.current });

            hotKeysPressed.current = [];

            event.preventDefault();
        }
    });

    return (
        <TextField
            {...props}
            value={value}
            ref={textFieldRef}
            placeholder={'Press key(s)'}
            onFocusChange={onFocusChange}
        />
    );
};
