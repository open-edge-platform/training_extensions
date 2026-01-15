// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, Grid } from '@geti/ui';

import { AnnotatorTools } from '../../../annotator/tools/annotator-tools/annotator-tools.component';
import { Hotkeys } from './hotkeys/hotkeys.component';
import { Settings } from './settings/settings.component';
import { ToggleAnnotationsVisibility } from './toggle-annotations-visibility.component';
import { ToggleFocus } from './toggle-focus.component';
import { UndoRedo } from './undo-redo/undo-redo.component';
import { ZoomFitScreen } from './zoom/zoom-fit-screen.component';
import { ZoomSelector } from './zoom/zoom-selector.component';

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
                    <ToggleAnnotationsVisibility />

                    <Settings />

                    <Divider size='S' />

                    <ZoomSelector />

                    <Divider size='S' />

                    <ToggleFocus />

                    <ZoomFitScreen />
                </Flex>

                <Flex UNSAFE_className={classes.toolbarSection} direction={'column'}>
                    <Hotkeys />
                </Flex>
            </Grid>
        </Flex>
    );
};
