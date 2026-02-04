// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { KeyboardEvent } from 'react';

import { TextField } from '@geti/ui';

import { formatHotkeyForDisplay } from '../../shared/hotkeys-definition';
import { validateLabelHotkey } from './label-validation';

interface HotkeyFieldProps {
    hotkey: string | null | undefined;
    onHotkeyChange: (hotkey: string | null) => void;
    allHotkeys: string[];
}

export const HotkeyField = ({ hotkey, onHotkeyChange, allHotkeys }: HotkeyFieldProps) => {
    const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
        event.preventDefault();

        const { key, ctrlKey, altKey, shiftKey, metaKey } = event;

        // Ignore standalone modifier keys
        if (['Control', 'Alt', 'Shift', 'Meta'].includes(key)) {
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

    const validationResult = hotkey == null ? undefined : validateLabelHotkey(hotkey, allHotkeys);
    const formattedHotkey = hotkey == null ? '' : formatHotkeyForDisplay(hotkey);

    return (
        <TextField
            aria-label={'Hotkey input'}
            placeholder={'Hotkey'}
            value={formattedHotkey}
            onKeyDown={handleKeyDown}
            width={'100%'}
            errorMessage={validationResult}
            validationState={validationResult === undefined ? undefined : 'invalid'}
        />
    );
};
