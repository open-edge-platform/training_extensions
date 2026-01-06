// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Flex } from '@geti/ui';
import { Redo, Undo } from '@geti/ui/icons';

import { useUndoRedo } from './undo-redo-provider.component';

export const UndoRedo = ({ isDisabled }: { isDisabled?: boolean }) => {
    const { undo, canUndo, redo, canRedo } = useUndoRedo();

    // TODO: add 'react-hotkeys-hook' package
    // useHotkeys(HOTKEYS.undo, undo, [undo]);
    // useHotkeys(HOTKEYS.redo, redo, [redo]);

    return (
        <Flex alignItems='center' direction={'column'} justifyContent={'center'} data-testid='undo-redo-tools'>
            <ActionButton
                isQuiet
                id='undo-button'
                data-testid='undo-button'
                onPress={undo}
                aria-label='undo'
                isDisabled={!canUndo || isDisabled}
            >
                <Undo />
            </ActionButton>

            <ActionButton
                isQuiet
                id='redo-button'
                data-testid='redo-button'
                aria-label='redo'
                onPress={redo}
                isDisabled={!canRedo || isDisabled}
            >
                <Redo />
            </ActionButton>
        </Flex>
    );
};
