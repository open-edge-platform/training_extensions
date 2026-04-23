// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Button, ButtonGroup, Divider, Flex, Icon, Text } from '@geti/ui';
import { CloseSemiBold } from '@geti/ui/icons';
import { useProject } from 'hooks/api/project.hook';
import { isEmpty } from 'lodash-es';

import type { DatasetSubset, Media } from '../../../../constants/shared-types';
import { useAnnotationActions } from '../../../../shared/annotator/annotation-actions-provider.component';
import type { AnnotatorMode } from '../../../../shared/annotator/annotator-mode';
import { isVideoFrame } from '../../../../shared/media-item-utils';
import { Labels } from '../../../annotator/labels/labels.component';
import { useVideoPlayerContext } from '../../../annotator/video-player/video-player-provider.component';
import { isClassificationTask, isMultiLabelClassificationTask } from '../../../project/task-type-guards';
import { DeleteMediaItem } from '../../gallery/delete-media-item/delete-media-item.component';
import { Toolbar } from '../toolbar-container/toolbar-container.component';
import { AnnotatorModes } from './annotator-modes/annotator-modes-toggle.component';
import { PredictionModelSelector } from './annotator-modes/prediction-model-selector.component';
import { PredictionButtons } from './annotator-modes/predictions-buttons.component';
import { getNextItem } from './util';

import classes from './secondary-toolbar.module.scss';

type AnnotationButtonsProps = {
    onDeleteItem: ([deletedItem]: string[]) => void;
    mediaId: string;
    onSubmit: () => void;
    isDisabled: boolean;
    isSaving: boolean;
};

const AnnotationButtons = ({ onDeleteItem, mediaId, onSubmit, isDisabled, isSaving }: AnnotationButtonsProps) => {
    return (
        <>
            <DeleteMediaItem itemsIds={[mediaId]} onDeleted={onDeleteItem} />
            <Button variant='accent' onPress={onSubmit} isPending={isSaving} isDisabled={isDisabled}>
                Submit
            </Button>
        </>
    );
};

type SecondaryToolbarProps = {
    items: Media[];
    mediaItem: Media;
    mode: AnnotatorMode;
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;
    onModeChange: (mode: AnnotatorMode) => void;
    onSelectNextMediaItem: () => void;
    subset: DatasetSubset;
    hasSubsetChanged: boolean;
    isLoadingPredictions: boolean;
};

export const SecondaryToolbar = ({
    items,
    mediaItem,
    mode,
    onClose,
    onSelectedMediaItem,
    onModeChange,
    onSelectNextMediaItem,
    subset,
    hasSubsetChanged = false,
    isLoadingPredictions = false,
}: SecondaryToolbarProps) => {
    const { data: selectedProject } = useProject();
    const videoPlayerContext = useVideoPlayerContext();
    const isPlaying = videoPlayerContext?.videoControls?.isPlaying ?? false;

    const { canSubmit, isSaving, submitAnnotations, initialAnnotations, initialPredictions } = useAnnotationActions();

    const handleSubmit = async () => {
        await submitAnnotations(subset);
        onSelectNextMediaItem();
    };

    const isClassification = isClassificationTask(selectedProject.task.task_type);
    const isMultiLabelClassification = isMultiLabelClassificationTask(selectedProject.task);

    const handleDeleteItem = ([deletedItem]: string[], totalItems: number) => {
        const deletedIndex = items.findIndex((item) => item.id === deletedItem);
        const nextItem = getNextItem(totalItems - 1, deletedIndex);

        onSelectedMediaItem(items[nextItem]);
    };

    const isPredictionMode = mode === 'prediction';
    const isAnnotationMode = mode === 'annotation';

    // If annotations are not changed but subset has changed we want to allow user to submit
    const isSubmitDisabled = (!canSubmit && !hasSubsetChanged) || isSaving || isLoadingPredictions;

    return (
        <Flex
            width={'100%'}
            height={'100%'}
            alignItems={'center'}
            justifyContent={'space-between'}
            UNSAFE_className={classes.secondaryToolbarContainer}
        >
            <Toolbar.Container>
                <Toolbar.Section>
                    <Flex alignItems={'center'} gap={'size-200'}>
                        <AnnotatorModes
                            // We want to reset annotation and/or prediction cue when media item changes
                            key={isVideoFrame(mediaItem) ? `${mediaItem.id}-${mediaItem.frame_number}` : mediaItem.id}
                            mode={mode}
                            onModeChange={onModeChange}
                            hasAnnotations={!isEmpty(initialAnnotations)}
                            hasPredictions={!isEmpty(initialPredictions)}
                        />
                        {isPredictionMode && <PredictionModelSelector isDisabled={isLoadingPredictions || isPlaying} />}
                    </Flex>
                </Toolbar.Section>
            </Toolbar.Container>
            {isAnnotationMode && (
                <Toolbar.Container>
                    <Toolbar.Section>
                        <Labels isClassification={isClassification} isMultiLabel={isMultiLabelClassification} />
                    </Toolbar.Section>
                </Toolbar.Container>
            )}
            <Toolbar.Container>
                <Toolbar.Section>
                    <ButtonGroup UNSAFE_className={classes.buttonsGroup}>
                        {isPredictionMode && (
                            <PredictionButtons
                                onModeChange={onModeChange}
                                isDisabled={isSubmitDisabled}
                                onSubmit={handleSubmit}
                            />
                        )}
                        {isAnnotationMode && (
                            <AnnotationButtons
                                mediaId={mediaItem.id}
                                onDeleteItem={(deleteItems) => handleDeleteItem(deleteItems, items.length - 1)}
                                onSubmit={handleSubmit}
                                isSaving={isSaving}
                                isDisabled={isSubmitDisabled}
                            />
                        )}

                        <Divider size={'S'} height={'size-400'} width={'size-10'} />

                        <ActionButton isQuiet onPress={onClose} isDisabled={isSaving}>
                            <Icon height={'size-150'} width={'size-150'}>
                                <CloseSemiBold />
                            </Icon>
                            <Text>Close</Text>
                        </ActionButton>
                    </ButtonGroup>
                </Toolbar.Section>
            </Toolbar.Container>
        </Flex>
    );
};
