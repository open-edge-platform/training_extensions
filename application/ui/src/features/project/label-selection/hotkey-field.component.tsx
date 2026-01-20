// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { KeyboardEvent } from 'react';

import { TextField } from '@geti/ui';

type HotkeyFieldProps = {
    hotkey: string | null | undefined;
    onHotkeyChange: (hotkey: string | null) => void;
};

export const HotkeyField = ({ hotkey, onHotkeyChange }: HotkeyFieldProps) => {
    const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
        event.preventDefault();

        const { key, ctrlKey, altKey, shiftKey, metaKey } = event;

        // Ignore standalone modifier keys
        if (['Control', 'Alt', 'Shift', 'Meta'].includes(key)) {
            return;
        }

        // Build the hotkey string
        const modifiers: string[] = [];
        if (ctrlKey) modifiers.push('ctrl');
        if (metaKey) modifiers.push('cmd');
        if (altKey) modifiers.push('alt');
        if (shiftKey) modifiers.push('shift');

        const keyUpperCased = key.toUpperCase();

        const hotkeyString = modifiers.length > 0 ? `${modifiers.join('+')}+${keyUpperCased}` : keyUpperCased;

        onHotkeyChange(hotkeyString);
    };

    return (
        <TextField
            aria-label={'Hotkey input'}
            placeholder={'Hotkey'}
            value={hotkey?.toUpperCase() ?? undefined}
            onKeyDown={handleKeyDown}
            width={'100%'}
        />
    );
};
