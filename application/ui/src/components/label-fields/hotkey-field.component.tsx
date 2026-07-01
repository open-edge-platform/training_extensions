// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { KeyboardEvent } from 'react';

import { TextField } from '@geti/ui';

import { formatHotkeyForDisplay } from '../../shared/hotkeys-definition';

type HotkeyFieldProps = {
    hotkey: string | null | undefined;
    onEnter?: () => void;
    onHotkeyChange: (hotkey: string | null) => void;
    errorMessage?: string;
};

const isEnter = (event: KeyboardEvent) => {
    return event.key === 'Enter';
};

export const HotkeyField = ({ hotkey, errorMessage, onEnter, onHotkeyChange }: HotkeyFieldProps) => {
    const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
        event.preventDefault();

        const { key, ctrlKey, altKey, shiftKey, metaKey } = event;

        // Ignore standalone modifier keys
        if (['Control', 'Alt', 'Shift', 'Meta', 'Enter'].includes(key)) {
            isEnter(event) && onEnter?.();

            return;
        }

        const modifiers: string[] = [];
        if (ctrlKey) modifiers.push('ctrl');
        if (metaKey) modifiers.push('meta');
        if (altKey) modifiers.push('alt');
        if (shiftKey) modifiers.push('shift');

        const hotkeyString = modifiers.length > 0 ? `${modifiers.join('+')}+${key}` : key;

        onHotkeyChange(hotkeyString);
    };

    const formattedHotkey = hotkey == null ? '' : formatHotkeyForDisplay(hotkey);

    return (
        <TextField
            aria-label={'Hotkey input'}
            placeholder={'Hotkey'}
            value={formattedHotkey}
            onKeyDown={handleKeyDown}
            width={'100%'}
            errorMessage={errorMessage}
            validationState={errorMessage ? 'invalid' : undefined}
        />
    );
};
