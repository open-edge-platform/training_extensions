// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Grid, Keyboard, Text } from '@geti/ui';

import { formatHotkeyForDisplay, HOTKEYS } from '../../../../../shared/hotkeys-definition';
import { useAvailableTools } from '../../../../annotator/tools/annotator-tools/use-available-tools';

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

export const HotkeysList = () => {
    const availableTools = useAvailableTools();

    return (
        <Grid columns={['2fr', '1fr']} rowGap={'size-100'}>
            {availableTools.map((tool) => (
                <HotkeyItem key={tool.label} hotkeyName={tool.label} hotkey={formatHotkeyForDisplay(tool.hotkey)} />
            ))}
            <Divider size='S' gridColumn={'1/-1'} />
            <HotkeyItem hotkeyName={'Undo'} hotkey={formatHotkeyForDisplay(HOTKEYS.undo)} />
            <HotkeyItem hotkeyName={'Redo'} hotkey={formatHotkeyForDisplay(HOTKEYS.redo)} />
            <HotkeyItem
                hotkeyName={'Delete selected annotation'}
                hotkey={formatHotkeyForDisplay(HOTKEYS.deleteAnnotation)}
            />
            <HotkeyItem
                hotkeyName={'Show or hide all annotations'}
                hotkey={formatHotkeyForDisplay(HOTKEYS.toggleAnnotationsVisibility)}
            />
            <HotkeyItem
                hotkeyName={'Select all annotations'}
                hotkey={formatHotkeyForDisplay(HOTKEYS.selectAllAnnotations)}
            />
            <HotkeyItem
                hotkeyName={'Deselect all annotations'}
                hotkey={formatHotkeyForDisplay(HOTKEYS.deselectAllAnnotations)}
            />
            <Divider size='S' gridColumn={'1/-1'} />
            <HotkeyItem hotkeyName={'Reset zoom'} hotkey={formatHotkeyForDisplay(HOTKEYS.fitToScreen)} />
        </Grid>
    );
};
