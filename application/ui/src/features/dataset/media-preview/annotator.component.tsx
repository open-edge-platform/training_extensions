// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';

import type { Media } from '../../../constants/shared-types';
import { isVideo, isVideoFrame } from '../../../shared/media-item-utils';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas/annotator-canvas';
import { useSelectedMediaItem } from '../../annotator/selected-media-item-provider.component';
import { VideoPlayerProvider } from '../../annotator/video-player/video-player-provider.component';
import { VideoToolbar } from '../../annotator/video-player/video-toolbar/video-toolbar.component';
import { BottomToolbar } from './bottom-toolbar/bottom-toolbar.component';
import { PrimaryToolbar } from './primary-toolbar/primary-toolbar.component';
import { AnnotatorCanvasSettings } from './primary-toolbar/settings/annotator-canvas-settings.component';
import { ReadOnlyAnnotator } from './read-only-annotator.component';
import { AnnotatorMode } from './secondary-toolbar/annotator-modes/mode';
import { SecondaryToolbar } from './secondary-toolbar/secondary-toolbar.component';
import { useNextMediaPrefetch } from './utils';

type AnnotatorProps = {
    image: ImageData;
    mediaItem: Media;
    items: Media[];
    mode: AnnotatorMode;
    changeAnnotatorMode: (mode: AnnotatorMode) => void;
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;
};

const Annotator = ({
    mediaItem,
    image,
    mode,
    changeAnnotatorMode,
    onSelectedMediaItem,
    items,
    onClose,
}: AnnotatorProps) => {
    const { nextMediaItem } = useNextMediaPrefetch(mediaItem, items);

    const handleSubmitAnnotations = async () => {
        if (nextMediaItem === undefined) {
            return;
        }

        onSelectedMediaItem(nextMediaItem);
    };

    if (mode === 'prediction') {
        return (
            <ReadOnlyAnnotator
                mode={mode}
                image={image}
                mediaItem={mediaItem}
                onModeChange={changeAnnotatorMode}
                onClose={onClose}
                onAcceptPrediction={handleSubmitAnnotations}
            />
        );
    }

    return (
        <>
            <View gridArea={'header'}>
                <SecondaryToolbar
                    mode={mode}
                    items={items}
                    onClose={onClose}
                    mediaItem={mediaItem}
                    onSelectedMediaItem={onSelectedMediaItem}
                    onModeChange={changeAnnotatorMode}
                    onAcceptPrediction={handleSubmitAnnotations}
                />
            </View>

            <View gridArea={'toolbar'} aria-label={'primary toolbar'}>
                <PrimaryToolbar />
            </View>

            {(isVideo(mediaItem) || isVideoFrame(mediaItem)) && (
                <View gridArea={'video-toolbar'}>
                    <VideoToolbar mode={mode} />
                </View>
            )}

            <View gridArea={'bottom'}>
                <BottomToolbar mediaItem={mediaItem} />
            </View>

            <View gridArea={'canvas'} overflow={'hidden'}>
                <AnnotatorCanvasSettings>
                    <AnnotatorCanvas mediaItem={mediaItem} image={image} mode={mode} />
                </AnnotatorCanvasSettings>
            </View>
        </>
    );
};

type AnnotatorContainerProps = {
    mode: AnnotatorMode;
    changeAnnotatorMode: (mode: AnnotatorMode) => void;
    onClose: () => void;
    items: Media[];
    onSelectedMediaItem: (item: Media) => void;
};

export const AnnotatorContainer = ({
    mode,
    changeAnnotatorMode,
    onClose,
    items,
    onSelectedMediaItem,
}: AnnotatorContainerProps) => {
    const { mediaItem, setMediaItem, image } = useSelectedMediaItem();

    const selectMediaItem = (item: Media) => {
        setMediaItem(item);
        onSelectedMediaItem(item);
    };

    return (
        <VideoPlayerProvider
            videoFrame={isVideoFrame(mediaItem) ? mediaItem : undefined}
            changeSelectedMediaItem={selectMediaItem}
        >
            <Annotator
                mode={mode}
                items={items}
                image={image}
                onClose={onClose}
                mediaItem={mediaItem}
                changeAnnotatorMode={changeAnnotatorMode}
                onSelectedMediaItem={selectMediaItem}
            />
        </VideoPlayerProvider>
    );
};
