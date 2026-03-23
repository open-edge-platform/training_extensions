// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Button, Flex, Text, View } from '@geti/ui';
import { Checkmark, CloseSemiBold } from '@geti/ui/icons';

import type { Media } from '../../../constants/shared-types';
import type { AnnotatorMode } from '../../../shared/annotator/annotator-mode';
import { isVideo, isVideoFrame } from '../../../shared/media-item-utils';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas/annotator-canvas';
import { PredictionsSetupProvider } from '../../annotator/predictions-setup-provider.component';
import { VideoToolbar } from '../../annotator/video-player/video-toolbar/video-toolbar.component';
import { BottomToolbar } from './bottom-toolbar/bottom-toolbar.component';
import { AnnotatorCanvasSettings } from './primary-toolbar/settings/annotator-canvas-settings.component';
import { AnnotatorModes } from './secondary-toolbar/annotator-modes/annotator-modes-toggle.component';
import { PredictionModelSelector } from './secondary-toolbar/annotator-modes/prediction-model-selector.component';
import { Toolbar } from './toolbar-container/toolbar-container.component';
import { useSubmitPredictions } from './use-submit-predictions.hook';

import classes from './read-only-annotator.module.scss';

type ReadOnlyAnnotatorProps = {
    mode: AnnotatorMode;
    mediaItem: Media;
    image: ImageData;
    onClose: () => void;
    onModeChange?: (mode: 'annotation' | 'prediction') => void;
    onAcceptPrediction?: () => void;
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
    onAcceptPrediction,
}: ReadOnlyAnnotatorProps) => {
    const { canSubmit, isSaving, submit } = useSubmitPredictions({ onSuccess: onAcceptPrediction });

    return (
        <PredictionsSetupProvider>
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
                            {onModeChange && (
                                <Button
                                    variant='accent'
                                    onPress={submit}
                                    isPending={isSaving}
                                    marginStart={'size-200'}
                                    isDisabled={!canSubmit || isSaving}
                                >
                                    <Checkmark />
                                    <Text>Confirm prediction</Text>
                                </Button>
                            )}

                            <ActionButton isQuiet onPress={onClose} UNSAFE_className={classes.closeButton}>
                                <CloseSemiBold width={14} height={14} />
                                <Text>Close</Text>
                            </ActionButton>
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
        </PredictionsSetupProvider>
    );
};
