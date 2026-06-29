// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef, useState } from 'react';

import { Key, View } from '@geti-ui/ui';
import { useSpinDelay } from 'spin-delay';

import type { DatasetSubset, Media } from '../../../constants/shared-types';
import type { AnnotatorMode } from '../../../shared/annotator/annotator-mode';
import { isVideo, isVideoFrame } from '../../../shared/media-item-utils';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas/annotator-canvas';
import {
    useIsFetchingCurrentRangeFramesPredictions,
    useIsFetchingPredictions,
} from '../../annotator/api/use-media-predictions';
import { useSelectedMediaItem } from '../../annotator/selected-media-item-provider.component';
import { ToolManagerProvider } from '../../annotator/tools/tool-manager-provider.component';
import { VideoPlayerProvider } from '../../annotator/video-player/video-player-provider.component';
import { VideoToolbar } from '../../annotator/video-player/video-toolbar/video-toolbar.component';
import { BottomToolbar } from './bottom-toolbar/bottom-toolbar.component';
import { PrimaryToolbar } from './primary-toolbar/primary-toolbar.component';
import { AnnotatorCanvasSettings } from './primary-toolbar/settings/annotator-canvas-settings.component';
import { SecondaryToolbar } from './secondary-toolbar/secondary-toolbar.component';
import { useNextPredictionPrefetch } from './use-next-prediction-prefetch.hook';
import { useNextMediaPrefetch, usePlayPauseVideoBySystem } from './utils';

const DATASET_SUBSETS: DatasetSubset[] = ['unassigned', 'training', 'validation', 'testing'];

const isDatasetSubset = (key: Key | null): key is DatasetSubset => DATASET_SUBSETS.includes(key as DatasetSubset);

const constructMediaItemId = (mediaItem: Media): string => {
    return isVideoFrame(mediaItem) ? `${mediaItem.id}-${mediaItem.frame_number}` : mediaItem.id;
};

const useSubset = (subset: DatasetSubset, mediaItem: Media) => {
    const [pendingSubset, setPendingSubset] = useState<DatasetSubset>(subset);
    const prevMediaItemIdRef = useRef<string>(constructMediaItemId(mediaItem));
    const currentMediaItemId = constructMediaItemId(mediaItem);
    const prevSubsetRef = useRef(subset);

    if (prevMediaItemIdRef.current !== currentMediaItemId) {
        prevMediaItemIdRef.current = currentMediaItemId;
        setPendingSubset(subset);
    }

    if (prevSubsetRef.current !== subset) {
        prevSubsetRef.current = subset;
        setPendingSubset(subset);
    }

    const changeSubset = (key: Key | null) => {
        if (isDatasetSubset(key)) {
            setPendingSubset(key);
        }
    };

    return {
        currentSubset: pendingSubset,
        changeCurrentSubset: changeSubset,
        isReadOnlySubset: pendingSubset === subset && subset !== 'unassigned',
    };
};

type AnnotatorProps = {
    image: ImageData;
    mediaItem: Media;
    items: Media[];
    mode: AnnotatorMode;
    onChangeAnnotatorMode: (mode: AnnotatorMode) => void;
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;
    subset: DatasetSubset;
    isUserReviewed: boolean;
};

const NextPredictionPrefetch = ({ nextMediaItem }: { nextMediaItem: Media }) => {
    useNextPredictionPrefetch(nextMediaItem);

    return null;
};

const Annotator = ({
    mediaItem,
    image,
    mode,
    items,
    onClose,
    subset,
    isUserReviewed,
    onSelectedMediaItem,
    onChangeAnnotatorMode,
}: AnnotatorProps) => {
    const isAnnotationMode = mode === 'annotation';
    const isPredictionMode = mode === 'prediction';

    const { nextMediaItem } = useNextMediaPrefetch(mediaItem, items);
    const { currentSubset, changeCurrentSubset, isReadOnlySubset } = useSubset(subset, mediaItem);
    const isLoadingPredictions = useIsFetchingPredictions(mediaItem.id) && isPredictionMode;
    const isLoadingCurrentRangePredictions =
        useIsFetchingCurrentRangeFramesPredictions(mediaItem.id) && isPredictionMode;

    const isLoadingFramesPredictionsDelayed = useSpinDelay(isLoadingPredictions, {
        delay: 400,
        minDuration: 200,
    });

    usePlayPauseVideoBySystem(isLoadingCurrentRangePredictions);

    const selectNextMediaItem = async () => {
        if (nextMediaItem === undefined) {
            return;
        }

        onSelectedMediaItem(nextMediaItem);
    };

    return (
        <>
            {isPredictionMode && nextMediaItem && <NextPredictionPrefetch nextMediaItem={nextMediaItem} />}
            <View gridArea={'header'}>
                <SecondaryToolbar
                    mode={mode}
                    items={items}
                    onClose={onClose}
                    mediaItem={mediaItem}
                    onSelectedMediaItem={onSelectedMediaItem}
                    onModeChange={onChangeAnnotatorMode}
                    onSelectNextMediaItem={selectNextMediaItem}
                    subset={currentSubset}
                    hasSubsetChanged={currentSubset !== subset}
                    isLoadingPredictions={isLoadingPredictions}
                />
            </View>

            {isAnnotationMode && (
                <View gridArea={'toolbar'} aria-label={'primary toolbar'}>
                    <PrimaryToolbar />
                </View>
            )}

            {(isVideo(mediaItem) || isVideoFrame(mediaItem)) && (
                <View gridArea={'video-toolbar'}>
                    <VideoToolbar mode={mode} />
                </View>
            )}

            <View gridArea={'bottom'}>
                <BottomToolbar
                    isUserReviewed={isUserReviewed}
                    subset={currentSubset}
                    onSubsetChange={changeCurrentSubset}
                    mediaItem={mediaItem}
                    isReadOnlySubset={isReadOnlySubset}
                />
            </View>

            <View gridArea={'canvas'} overflow={'hidden'} position={'relative'}>
                <AnnotatorCanvasSettings>
                    <AnnotatorCanvas
                        isLoadingPredictions={isLoadingFramesPredictionsDelayed}
                        mediaItem={mediaItem}
                        image={image}
                        mode={mode}
                        isReadOnly={isPredictionMode}
                    />
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
    subset: DatasetSubset;
    isUserReviewed: boolean;
};

export const AnnotatorContainer = ({
    mode,
    changeAnnotatorMode,
    onClose,
    items,
    subset,
    isUserReviewed,
    onSelectedMediaItem,
}: AnnotatorContainerProps) => {
    const { mediaItem, image } = useSelectedMediaItem();

    return (
        <VideoPlayerProvider
            videoFrame={isVideoFrame(mediaItem) ? mediaItem : undefined}
            changeSelectedMediaItem={onSelectedMediaItem}
        >
            <ToolManagerProvider>
                <Annotator
                    mode={mode}
                    subset={subset}
                    items={items}
                    image={image}
                    onClose={onClose}
                    mediaItem={mediaItem}
                    isUserReviewed={isUserReviewed}
                    onChangeAnnotatorMode={changeAnnotatorMode}
                    onSelectedMediaItem={onSelectedMediaItem}
                />
            </ToolManagerProvider>
        </VideoPlayerProvider>
    );
};
