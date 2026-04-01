// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Key, View } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../api/client';
import type { DatasetSubset, Media } from '../../../constants/shared-types';
import type { AnnotatorMode } from '../../../shared/annotator/annotator-mode';
import { isVideo, isVideoFrame } from '../../../shared/media-item-utils';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas/annotator-canvas';
import { useSelectedMediaItem } from '../../annotator/selected-media-item-provider.component';
import { VideoPlayerProvider } from '../../annotator/video-player/video-player-provider.component';
import { VideoToolbar } from '../../annotator/video-player/video-toolbar/video-toolbar.component';
import { BottomToolbar } from './bottom-toolbar/bottom-toolbar.component';
import { PrimaryToolbar } from './primary-toolbar/primary-toolbar.component';
import { AnnotatorCanvasSettings } from './primary-toolbar/settings/annotator-canvas-settings.component';
import { SecondaryToolbar } from './secondary-toolbar/secondary-toolbar.component';
import { useNextMediaPrefetch } from './utils';

const DATASET_SUBSETS: DatasetSubset[] = ['unassigned', 'training', 'validation', 'testing'];
const isDatasetSubset = (key: Key | null): key is DatasetSubset => DATASET_SUBSETS.includes(key as DatasetSubset);

const useSubsets = (mediaItem: Media) => {
    const projectId = useProjectIdentifier();
    const [pendingSubset, setPendingSubset] = useState<DatasetSubset | null>(null);
    const [prevMediaItemId, setPrevMediaItemId] = useState(mediaItem.id);

    if (prevMediaItemId !== mediaItem.id) {
        setPrevMediaItemId(mediaItem.id);
        setPendingSubset(null);
    }

    const datasetItemParams = { params: { path: { project_id: projectId, dataset_item_id: mediaItem.id } } };

    const { data } = $api.useQuery(
        'get',
        '/api/projects/{project_id}/dataset/items/{dataset_item_id}',
        datasetItemParams
    );

    const handleSubsetChange = (key: Key | null) => {
        if (isDatasetSubset(key)) {
            setPendingSubset(key);
        }
    };

    const currentSubset: DatasetSubset = data?.subset ?? 'unassigned';
    const subset: DatasetSubset = pendingSubset ?? currentSubset;

    return {
        isUserReviewed: data?.user_reviewed ?? false,
        subset,
        handleSubsetChange,
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
    const { isUserReviewed, subset, handleSubsetChange } = useSubsets(mediaItem);

    const selectNextMediaItem = async () => {
        if (nextMediaItem === undefined) {
            return;
        }

        onSelectedMediaItem(nextMediaItem);
    };

    const isAnnotationMode = mode === 'annotation';
    const isPredictionMode = mode === 'prediction';

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
                    onSelectNextMediaItem={selectNextMediaItem}
                    subset={subset}
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
                    subset={subset}
                    handleSubsetChange={handleSubsetChange}
                    mediaItem={mediaItem}
                />
            </View>

            <View gridArea={'canvas'} overflow={'hidden'}>
                <AnnotatorCanvasSettings>
                    <AnnotatorCanvas mediaItem={mediaItem} image={image} mode={mode} isReadOnly={isPredictionMode} />
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
