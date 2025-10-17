// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, dimensionValue, Flex, Grid } from '@geti/ui';
import { isEmpty } from 'lodash-es';
import { useAnnotationActions } from 'src/features/annotator/annotation-actions-provider.component';

import { LabelPicker } from './label-picker.component';
import { useSecondaryToolbarState } from './use-secondary-toolbar-state.hook';

import classes from '../media-preview.module.scss';

type SecondaryToolbarProps = {
    onClose: () => void;
};

export const SecondaryToolbar = ({ onClose }: SecondaryToolbarProps) => {
    const { annotations, isSaving, isUserReviewed, submitAnnotations } = useAnnotationActions();
    const { isHidden, projectLabels, toggleLabels, annotationsToUpdate } = useSecondaryToolbarState();

    const annotationLabelId = annotationsToUpdate.at(0)?.labels?.at(0)?.id;
    const selectedLabel = projectLabels.find((label) => label.id === annotationLabelId) ?? null;
    const hasAnnotations = !isEmpty(annotations);

    const handleAccept = async () => {
        await submitAnnotations();
    };

    return (
        <Flex
            height={'100%'}
            width={'100%'}
            alignItems={'center'}
            UNSAFE_style={{ paddingTop: dimensionValue('size-125') }}
        >
            <Grid width={'100%'} UNSAFE_className={classes.toolbarGrid} isHidden={isHidden}>
                <Flex UNSAFE_className={classes.toolbarSection} justifyContent={'space-between'}>
                    <LabelPicker selectedLabel={selectedLabel} labels={projectLabels} onSelect={toggleLabels} />
                    <ButtonGroup>
                        <Button
                            variant='accent'
                            onPress={handleAccept}
                            isPending={isSaving}
                            isDisabled={!hasAnnotations || isSaving}
                        >
                            {isUserReviewed ? 'Submit' : 'Approve'}
                        </Button>

                        <Button variant='secondary' onPress={onClose} isDisabled={isSaving}>
                            Close
                        </Button>
                    </ButtonGroup>
                </Flex>
            </Grid>
        </Flex>
    );
};
