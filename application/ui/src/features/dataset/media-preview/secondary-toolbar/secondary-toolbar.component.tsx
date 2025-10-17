// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, dimensionValue, Flex, Grid } from '@geti/ui';
import { QueryClient, useQueryClient } from '@tanstack/react-query';
import { isEmpty } from 'lodash-es';
import { useAnnotationActions } from 'src/features/annotator/annotation-actions-provider.component';
import { DatasetItem } from 'src/features/annotator/types';

import { LabelPicker } from './label-picker.component';
import { useSecondaryToolbarState } from './use-secondary-toolbar-state.hook';

import classes from '../media-preview.module.scss';

type SecondaryToolbarProps = {
    items: DatasetItem[];
    mediaItem: DatasetItem;
    onClose: () => void;
    onSelectedMediaItem: (item: DatasetItem) => void;
};

const getNextItem = (totalItems: number, newIndex: number) => {
    return Math.min(totalItems, newIndex + 1);
};

const invalidateMediaItemAnnotations = (queryClient: QueryClient) => {
    queryClient.invalidateQueries({
        queryKey: ['get', '/api/projects/{project_id}/dataset/items/{dataset_item_id}/annotations'],
    });
};

export const SecondaryToolbar = ({ items, mediaItem, onClose, onSelectedMediaItem }: SecondaryToolbarProps) => {
    const queryClient = useQueryClient();
    const { annotations, isSaving, isUserReviewed, submitAnnotations } = useAnnotationActions();
    const { isHidden, projectLabels, toggleLabels, annotationsToUpdate } = useSecondaryToolbarState();

    const annotationLabelId = annotationsToUpdate.at(0)?.labels?.at(0)?.id;
    const selectedLabel = projectLabels.find((label) => label.id === annotationLabelId) ?? null;
    const hasAnnotations = !isEmpty(annotations);
    const selectedIndex = items.findIndex((item) => item.id === mediaItem.id);

    const handleSubmit = async () => {
        await submitAnnotations();

        const nextItem = getNextItem(items.length - 1, selectedIndex);
        onSelectedMediaItem(items[nextItem]);

        const isLastItem = selectedIndex === items.length - 1;
        isLastItem && invalidateMediaItemAnnotations(queryClient);
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
                            onPress={handleSubmit}
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
