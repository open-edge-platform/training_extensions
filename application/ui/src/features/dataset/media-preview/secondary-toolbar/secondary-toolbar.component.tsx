// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Button, ButtonGroup, Divider, Flex, Icon, Text } from '@geti-ui/ui';
import { CloseSemiBold } from '@geti-ui/ui/icons';
import { useProject } from 'hooks/api/project.hook';
import { isEmpty } from 'lodash-es';
import { useHotkeys } from 'react-hotkeys-hook';

import type { DatasetSubset, Media } from '../../../../constants/shared-types';
import { useAnnotationActions } from '../../../../shared/annotator/annotation-actions-provider.component';
import type { AnnotatorMode } from '../../../../shared/annotator/annotator-mode';
import { HOTKEYS } from '../../../../shared/hotkeys-definition';
import { isImage, isVideoFrame } from '../../../../shared/media-item-utils';
import { Labels } from '../../../annotator/labels/labels.component';
import { usePredictionSetup } from '../../../annotator/predictions-setup-provider.component';
import { useVideoPlayerContext } from '../../../annotator/video-player/video-player-provider.component';
import { isClassificationTask, isMultiLabelClassificationTask } from '../../../project/task-type-guards';
import { DeleteMediaItem } from '../../gallery/delete-media-item/delete-media-item.component';
import { Toolbar } from '../toolbar-container/toolbar-container.component';
import { AnnotatorModes } from './annotator-modes/annotator-modes-toggle.component';
import { PredictionInferenceDevices } from './annotator-modes/prediction-inference-devices.component';
import { PredictionModelSelector } from './annotator-modes/prediction-model-selector.component';
import { PredictionButtons } from './annotator-modes/predictions-buttons.component';
import { getNextItem } from './util';

import classes from './secondary-toolbar.module.scss';

type ImageAnnotationButtonsProps = {
    onDeleteItem: ([deletedItem]: string[]) => void;
    mediaId: string;
    onSubmit: () => void;
    isDisabled: boolean;
    isSaving: boolean;
};

const ImageAnnotationButtons = ({
    onDeleteItem,
    mediaId,
    onSubmit,
    isDisabled,
    isSaving,
}: ImageAnnotationButtonsProps) => {
    return (
        <>
            <DeleteMediaItem itemsIds={[mediaId]} onDeleted={onDeleteItem} />
            <Button variant='accent' onPress={onSubmit} isPending={isSaving} isDisabled={isDisabled}>
                Submit
            </Button>
        </>
    );
};

type VideoAnnotationButtonsProps = {
    onSubmit: () => void;
    isDisabled: boolean;
    isSaving: boolean;
};

const VideoAnnotationButtons = ({ onSubmit, isDisabled, isSaving }: VideoAnnotationButtonsProps) => {
    return (
        <Button variant='accent' onPress={onSubmit} isPending={isSaving} isDisabled={isDisabled}>
            Submit
        </Button>
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
    const { selectableModels } = usePredictionSetup();
    const isPlaying = videoPlayerContext?.videoControls?.isPlaying ?? false;

    const { canSubmit, isSaving, submitAnnotations, submitPredictions, initialAnnotations, initialPredictions } =
        useAnnotationActions();

    const handleSubmit = async () => {
        await submitAnnotations(subset);
        onSelectNextMediaItem();
    };

    const handleSubmitPredictions = async () => {
        await submitPredictions(subset);
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

    useHotkeys(
        HOTKEYS.submit,
        (event) => {
            event.preventDefault();

            if (isPredictionMode) {
                handleSubmitPredictions();
            } else {
                handleSubmit();
            }
        },
        { enabled: !isSubmitDisabled },
        [isSubmitDisabled, isPredictionMode, handleSubmitPredictions, handleSubmit]
    );

    return (
        <Flex width={'100%'} height={'100%'} alignItems={'center'} justifyContent={'space-between'}>
            <Toolbar.Container>
                <Flex alignItems={'center'} gap={'size-50'}>
                    <Toolbar.Section>
                        <AnnotatorModes
                            // We want to reset annotation and/or prediction cue when media item changes
                            key={isVideoFrame(mediaItem) ? `${mediaItem.id}-${mediaItem.frame_number}` : mediaItem.id}
                            mode={mode}
                            onModeChange={onModeChange}
                            hasAnnotations={!isEmpty(initialAnnotations)}
                            hasPredictions={!isEmpty(initialPredictions)}
                        />
                    </Toolbar.Section>

                    {isPredictionMode && (
                        <Flex gap={'size-50'}>
                            {!isEmpty(selectableModels) ? (
                                <Toolbar.Section minWidth={'size-2000'}>
                                    <PredictionModelSelector isDisabled={isLoadingPredictions || isPlaying} />
                                </Toolbar.Section>
                            ) : null}
                            <Toolbar.Section>
                                <PredictionInferenceDevices isDisabled={isLoadingPredictions || isPlaying} />
                            </Toolbar.Section>
                        </Flex>
                    )}
                </Flex>
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
                                onSubmit={handleSubmitPredictions}
                            />
                        )}
                        {isAnnotationMode && isImage(mediaItem) && (
                            <ImageAnnotationButtons
                                mediaId={mediaItem.id}
                                onDeleteItem={(deleteItems) => handleDeleteItem(deleteItems, items.length - 1)}
                                onSubmit={handleSubmit}
                                isSaving={isSaving}
                                isDisabled={isSubmitDisabled}
                            />
                        )}
                        {isAnnotationMode && !isImage(mediaItem) && (
                            <VideoAnnotationButtons
                                onSubmit={handleSubmit}
                                isDisabled={isSubmitDisabled}
                                isSaving={isSaving}
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
