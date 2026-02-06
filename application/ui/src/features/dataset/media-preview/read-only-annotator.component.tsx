// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Button, Flex, Text, View } from '@geti/ui';
import { Checkmark, CloseSemiBold } from '@geti/ui/icons';

import type { Media } from '../../../constants/shared-types';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas/annotator-canvas';
import { BottomToolbar } from './bottom-toolbar/bottom-toolbar.component';
import { AnnotatorCanvasSettings } from './primary-toolbar/settings/annotator-canvas-settings.component';
import { AnnotatorModes } from './secondary-toolbar/annotator-modes/annotator-modes-toggle.component';
import { Toolbar } from './toolbar-container/toolbar-container.component';
import { useSubmitPredictions } from './use-submit-predictions.hook';

import classes from './read-only-annotator.module.scss';

type ReadOnlyAnnotatorProps = {
    mediaItem: Media;
    isUserReviewed: boolean;
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
    mediaItem,
    isUserReviewed,
    onModeChange,
    onClose,
    onAcceptPrediction,
}: ReadOnlyAnnotatorProps) => {
    const { canSubmit, isSaving, submit } = useSubmitPredictions({ onSuccess: onAcceptPrediction });

    return (
        <>
            <View gridArea={'header'} UNSAFE_className={classes.toolbarContainer}>
                <Flex alignItems={'center'} justifyContent={'space-between'} width={'100%'}>
                    {onModeChange ? (
                        <Toolbar.Container>
                            <Toolbar.Section>
                                <AnnotatorModes mode={'prediction'} onModeChange={onModeChange} />
                            </Toolbar.Section>
                        </Toolbar.Container>
                    ) : null}
                    <Toolbar.Container>
                        <Toolbar.Section>
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
                    <AnnotatorCanvas mediaItem={mediaItem} />
                </AnnotatorCanvasSettings>
            </View>

            <View gridArea={'bottom'}>
                <BottomToolbar isUserReviewed={isUserReviewed} mediaItem={mediaItem} hideHotkeys />
            </View>
        </>
    );
};
