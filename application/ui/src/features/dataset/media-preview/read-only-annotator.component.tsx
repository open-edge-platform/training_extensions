// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Divider, Flex, Icon, Text, View } from '@geti/ui';
import { Checkmark, CloseSemiBold, Edit } from '@geti/ui/icons';

import type { Media } from '../../../constants/shared-types';
import type { AnnotatorMode } from '../../../shared/annotator/annotator-mode';
import { isVideo, isVideoFrame } from '../../../shared/media-item-utils';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas/annotator-canvas';
import { VideoToolbar } from '../../annotator/video-player/video-toolbar/video-toolbar.component';
import { BottomToolbar } from './bottom-toolbar/bottom-toolbar.component';
import { AnnotatorCanvasSettings } from './primary-toolbar/settings/annotator-canvas-settings.component';
import { AnnotatorModes } from './secondary-toolbar/annotator-modes/annotator-modes-toggle.component';
import { PredictionModelSelector } from './secondary-toolbar/annotator-modes/prediction-model-selector.component';
import { Toolbar } from './toolbar-container/toolbar-container.component';
import { useSubmitPredictions } from './use-submit-predictions.hook';

import classes from './read-only-annotator.module.scss';

type EditPredictionButtonProps = {
    onEditPrediction: () => void;
};

const EditPredictionButton = ({ onEditPrediction }: EditPredictionButtonProps) => {
    return (
        <ActionButton isQuiet onPress={onEditPrediction}>
            <Icon>
                <Edit />
            </Icon>
            <Text>Edit</Text>
        </ActionButton>
    );
};

type ReadOnlyAnnotatorProps = {
    mode: AnnotatorMode;
    mediaItem: Media;
    image: ImageData;
    onClose: () => void;
    onModeChange?: (mode: AnnotatorMode) => void;
    onSuccessfulAcceptPrediction?: () => void;

    onEditPrediction?: () => void;
};

/**
 * Simplified read-only annotator for viewing predictions.
 *
 * Features:
 * - Read-only canvas (no annotation editing)
 * - Bottom toolbar without hotkeys
 * - No primary toolbar
 *
 * Note: This component renders into the parent grid layout from MediaPreview.
 * It uses the same gridArea structure as the normal annotator but with fewer elements.
 */
export const ReadOnlyAnnotator = ({
    mode,
    image,
    mediaItem,
    onModeChange,
    onClose,
    onSuccessfulAcceptPrediction,
    onEditPrediction,
}: ReadOnlyAnnotatorProps) => {
    const { canSubmit, isSaving, submit } = useSubmitPredictions({ onSuccess: onSuccessfulAcceptPrediction });

    return (
        <>
            <View gridArea={'header'} UNSAFE_className={classes.toolbarContainer}>
                <Flex alignItems={'center'} justifyContent={'space-between'} width={'100%'}>
                    {onModeChange && (
                        <Toolbar.Container>
                            <Toolbar.Section>
                                <Flex alignItems={'center'} gap={'size-200'}>
                                    <AnnotatorModes mode={'prediction'} onModeChange={onModeChange} />
                                    {mode === 'prediction' && <PredictionModelSelector />}
                                </Flex>
                            </Toolbar.Section>
                        </Toolbar.Container>
                    )}
                    <Toolbar.Container marginStart={!onModeChange ? 'auto' : undefined}>
                        <Toolbar.Section>
                            <Flex alignItems={'center'} height={'100%'} alignContent={'center'} gap={'size-150'}>
                                {onModeChange && (
                                    <ActionButton isQuiet onPress={submit} isDisabled={!canSubmit || isSaving}>
                                        <Checkmark />
                                        <Text>Confirm prediction</Text>
                                    </ActionButton>
                                )}

                                {onEditPrediction && <EditPredictionButton onEditPrediction={onEditPrediction} />}

                                {(onEditPrediction || onModeChange) && (
                                    <Divider size={'S'} height={'size-400'} width={'size-10'} />
                                )}

                                <ActionButton isQuiet onPress={onClose} UNSAFE_className={classes.closeButton}>
                                    <Icon height={'size-150'} width={'size-150'}>
                                        <CloseSemiBold />
                                    </Icon>
                                    <Text>Close</Text>
                                </ActionButton>
                            </Flex>
                        </Toolbar.Section>
                    </Toolbar.Container>
                </Flex>
            </View>

            <View gridArea={'canvas'} overflow={'hidden'} UNSAFE_className={classes.readOnlyCanvas}>
                <AnnotatorCanvasSettings>
                    <AnnotatorCanvas isReadOnly mediaItem={mediaItem} image={image} mode={mode} />
                </AnnotatorCanvasSettings>
            </View>

            {(isVideo(mediaItem) || isVideoFrame(mediaItem)) && (
                <View gridArea={'video-toolbar'}>
                    <VideoToolbar mode={mode} />
                </View>
            )}

            <View gridArea={'bottom'}>
                <BottomToolbar mediaItem={mediaItem} hideHotkeys />
            </View>
        </>
    );
};
