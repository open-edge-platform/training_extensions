// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, Grid } from '@geti/ui';

import { Hotkeys } from './hotkeys/hotkeys.component';
import { Settings } from './settings/settings.component';
import { Tools } from './tools/tools.component';
import { UndoRedo } from './undo-redo/undo-redo.component';

import classes from './tool-selection-bar.module.scss';

export const ToolSelectionBar = () => {
    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'center'}>
            <Grid UNSAFE_className={classes.grid}>
                <Flex UNSAFE_className={classes.section}>
                    <Tools />

                    <Divider size='S' />

                    <UndoRedo />
                </Flex>

                <Flex UNSAFE_className={classes.section}>
                    <Settings />
                </Flex>

                <Flex UNSAFE_className={classes.section}>
                    <Hotkeys />
                </Flex>
            </Grid>
        </Flex>
    );
};
