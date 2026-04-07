// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Flex, Icon, Text, View } from '@geti/ui';
import { CloseSemiBold } from '@geti/ui/icons';

import type { DatasetSubset, Media } from '../../../constants/shared-types';
import type { AnnotatorMode } from '../../../shared/annotator/annotator-mode';
import { isVideo, isVideoFrame } from '../../../shared/media-item-utils';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas/annotator-canvas';
import { VideoToolbar } from '../../annotator/video-player/video-toolbar/video-toolbar.component';
import { BottomToolbar } from './bottom-toolbar/bottom-toolbar.component';
import { AnnotatorCanvasSettings } from './primary-toolbar/settings/annotator-canvas-settings.component';
import { Toolbar } from './toolbar-container/toolbar-container.component';

import classes from './read-only-annotator.module.scss';

type ReadOnlyAnnotatorProps = {
    mode: AnnotatorMode;
    mediaItem: Media;
    image: ImageData;
    onClose: () => void;
    subset: DatasetSubset;
};

/**
 * Simplified read-only annotator for viewing annotations.
 *
 * Features:
 * - Read-only canvas (no annotation editing)
 * - Bottom toolbar without hotkeys
 * - No primary toolbar
 *
 * Note: This component renders into the parent grid layout from MediaPreview.
 * It uses the same gridArea structure as the normal annotator but with fewer elements.
 */
export const ReadOnlyAnnotator = ({ mode, image, mediaItem, onClose, subset }: ReadOnlyAnnotatorProps) => {
    return (
        <>
            <View gridArea={'header'} UNSAFE_className={classes.toolbarContainer}>
                <Flex alignItems={'center'} justifyContent={'space-between'} width={'100%'}>
                    <Toolbar.Container marginStart={'auto'}>
                        <Toolbar.Section>
                            <ActionButton isQuiet onPress={onClose}>
                                <Icon height={'size-150'} width={'size-150'}>
                                    <CloseSemiBold />
                                </Icon>
                                <Text>Close</Text>
                            </ActionButton>
                        </Toolbar.Section>
                    </Toolbar.Container>
                </Flex>
            </View>

            <View gridArea={'canvas'} overflow={'hidden'}>
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
                <BottomToolbar mediaItem={mediaItem} hideHotkeys subset={subset} isReadOnlySubset />
            </View>
        </>
    );
};
