// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, Grid } from '@geti/ui';

import { AnnotatorTools } from '../../../annotator/tools/annotator-tools/annotator-tools.component';
import { ToggleAnnotationsVisibility } from './toggle-annotations-visibility.component';
import { UndoRedo } from './undo-redo/undo-redo.component';

import classes from '../media-preview.module.scss';

export const PrimaryToolbar = () => {
    return (
        <Flex justifyContent={'center'}>
            <Grid UNSAFE_className={classes.toolbarGrid}>
                <Flex UNSAFE_className={classes.toolbarSection} direction={'column'}>
                    <AnnotatorTools />

                    <Divider size='S' />

                    <UndoRedo />
                </Flex>

                <Flex UNSAFE_className={classes.toolbarSection} direction={'column'}>
                    <ToggleAnnotationsVisibility />
                </Flex>
            </Grid>
        </Flex>
    );
};
