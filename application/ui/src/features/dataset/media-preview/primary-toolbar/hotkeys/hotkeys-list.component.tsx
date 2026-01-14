// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Grid, Keyboard, Text } from '@geti/ui';

import { COMMAND_KEY, HOTKEYS } from './hotkeys-definition';

interface HotkeyItemProps {
    hotkeyName: string;
    hotkey: string;
}

const HotkeyItem = ({ hotkeyName, hotkey }: HotkeyItemProps) => {
    return (
        <>
            <Text>{hotkeyName}</Text>
            <Keyboard>{hotkey}</Keyboard>
        </>
    );
};

export const getHotkey = (key: string): string => {
    if (key.includes(COMMAND_KEY)) {
        return key.replace(COMMAND_KEY, 'cmd').toLocaleUpperCase();
    }

    return key.toLocaleUpperCase();
};

export const HotkeysList = () => {
    return (
        <Grid columns={['2fr', '1fr']} rowGap={'size-100'}>
            <HotkeyItem hotkeyName={'Undo'} hotkey={getHotkey(HOTKEYS.undo)} />
            <HotkeyItem hotkeyName={'Redo'} hotkey={getHotkey(HOTKEYS.redo)} />
            <HotkeyItem hotkeyName={'Delete annotation'} hotkey={getHotkey(HOTKEYS.deleteAnnotation)} />
            <HotkeyItem hotkeyName={'Show or hide all annotations'} hotkey={getHotkey(HOTKEYS.toggleAnnotations)} />
            <HotkeyItem hotkeyName={'Reset zoom'} hotkey={getHotkey(HOTKEYS.fitToScreen)} />
        </Grid>
    );
};
