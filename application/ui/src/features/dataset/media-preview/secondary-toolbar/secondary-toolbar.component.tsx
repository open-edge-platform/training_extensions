// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid } from '@geti/ui';

import { LabelPicker } from './label-picker.component';
import { useSecondaryToolbarState } from './use-secondary-toolbar-state.hook';

import classes from '../media-preview.module.scss';

export const SecondaryToolbar = () => {
    const { isHidden, projectLabels, toggleLabels, annotationsToUpdate } = useSecondaryToolbarState();

    const annotationLabelId = annotationsToUpdate.at(0)?.labels?.at(0)?.id;
    const selectedLabel = projectLabels.find((label) => label.id === annotationLabelId) ?? null;

    return (
        <Flex height={'100%'} alignItems={'center'} margin={'size-100'} minHeight={'size-1200'}>
            <Grid UNSAFE_className={classes.toolbarGrid} isHidden={isHidden}>
                <Flex UNSAFE_className={classes.toolbarSection}>
                    <LabelPicker selectedLabel={selectedLabel} labels={projectLabels} onSelect={toggleLabels} />
                </Flex>
            </Grid>
        </Flex>
    );
};
