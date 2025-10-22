// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid } from '@geti/ui';

import { AnnotatorTools } from '../../../annotator/tools/annotator-tools.component';
import { Hotkeys } from './hotkeys/hotkeys.component';
import { Settings } from './settings/settings.component';
import { UndoRedo } from './undo-redo/undo-redo.component';

import classes from '../media-preview.module.scss';

export const PrimaryToolbar = () => {
    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'center'}>
            <Grid UNSAFE_className={classes.toolbarGrid}>
                <Flex UNSAFE_className={classes.toolbarSection} direction={'column'}>
                    <AnnotatorTools />

                    <UndoRedo />
                </Flex>

                <Flex UNSAFE_className={classes.toolbarSection} direction={'column'}>
                    <Settings />
                </Flex>

                <Flex UNSAFE_className={classes.toolbarSection} direction={'column'}>
                    <Hotkeys />
                </Flex>
            </Grid>
        </Flex>
    );
};
