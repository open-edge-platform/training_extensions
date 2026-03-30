// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import type { Media } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import type { AnnotatorMode } from '../../../shared/annotator/annotator-mode';
import { isVideo, isVideoFrame } from '../../../shared/media-item-utils';
import { convertPredictionToAnnotation } from '../../annotator/annotations/utils';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas/annotator-canvas';
import { useSelectedMediaItem } from '../../annotator/selected-media-item-provider.component';
import { VideoPlayerProvider } from '../../annotator/video-player/video-player-provider.component';
import { VideoToolbar } from '../../annotator/video-player/video-toolbar/video-toolbar.component';
import { BottomToolbar } from './bottom-toolbar/bottom-toolbar.component';
import { PrimaryToolbar } from './primary-toolbar/primary-toolbar.component';
import { AnnotatorCanvasSettings } from './primary-toolbar/settings/annotator-canvas-settings.component';
import { ReadOnlyAnnotator } from './read-only-annotator.component';
import { SecondaryToolbar } from './secondary-toolbar/secondary-toolbar.component';
import { useNextMediaPrefetch } from './utils';

type PredictionAnnotatorProps = {
    image: ImageData;
    mediaItem: Media;
    mode: AnnotatorMode;
    onChangeAnnotatorMode: (mode: AnnotatorMode) => void;
    onClose: () => void;
    onSuccessfulAcceptPrediction: () => void;
};

const PredictionAnnotator = ({
    mode,
    onChangeAnnotatorMode,
    mediaItem,
    image,
    onClose,
    onSuccessfulAcceptPrediction,
}: PredictionAnnotatorProps) => {
    const { replaceAnnotations, annotations } = useAnnotationActions();

    const handleEditPrediction = () => {
        onChangeAnnotatorMode('annotation');
        replaceAnnotations(annotations.map(convertPredictionToAnnotation));
    };

    return (
        <ReadOnlyAnnotator
            mode={mode}
            image={image}
            mediaItem={mediaItem}
            onModeChange={onChangeAnnotatorMode}
            onClose={onClose}
            onSuccessfulAcceptPrediction={onSuccessfulAcceptPrediction}
            onEditPrediction={handleEditPrediction}
            isEditPredictionDisabled={isEmpty(annotations)}
        />
    );
};

type AnnotatorProps = {
    image: ImageData;
    mediaItem: Media;
    items: Media[];
    mode: AnnotatorMode;
    onChangeAnnotatorMode: (mode: AnnotatorMode) => void;
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;
};

const Annotator = ({
    mediaItem,
    image,
    mode,
    onChangeAnnotatorMode,
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
            <PredictionAnnotator
                image={image}
                mediaItem={mediaItem}
                mode={mode}
                onChangeAnnotatorMode={onChangeAnnotatorMode}
                onClose={onClose}
                onSuccessfulAcceptPrediction={handleSubmitAnnotations}
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
                    onModeChange={onChangeAnnotatorMode}
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
    const { mediaItem, image } = useSelectedMediaItem();

    return (
        <VideoPlayerProvider
            videoFrame={isVideoFrame(mediaItem) ? mediaItem : undefined}
            changeSelectedMediaItem={onSelectedMediaItem}
        >
            <Annotator
                mode={mode}
                items={items}
                image={image}
                onClose={onClose}
                mediaItem={mediaItem}
                onChangeAnnotatorMode={changeAnnotatorMode}
                onSelectedMediaItem={onSelectedMediaItem}
            />
        </VideoPlayerProvider>
    );
};
