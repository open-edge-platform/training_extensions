// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid } from '@geti/ui';

import { Hotkeys } from '../../../components/tool-selection-bar/hotkeys/hotkeys.component';
import { Settings } from '../../../components/tool-selection-bar/settings/settings.component';
import { UndoRedo } from '../../../components/tool-selection-bar/undo-redo/undo-redo.component';
import { AnnotatorTools } from './annotator-tools.component';

import classes from './tool-selection-bar.module.scss';

export const ToolSelectionBar = () => {
    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'center'}>
            <Grid UNSAFE_className={classes.grid}>
                <Flex UNSAFE_className={classes.section}>
                    <AnnotatorTools />

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
