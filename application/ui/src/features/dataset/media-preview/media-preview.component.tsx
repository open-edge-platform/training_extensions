// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from 'react';

import { Content, Dialog, Grid, View } from '@geti/ui';
import { useGetDatasetMediaItems } from 'hooks/use-get-dataset-media-items.hook';

import type { Media } from '../../../constants/shared-types';
import type { AnnotatorMode } from '../../../shared/annotator/annotator-mode';
import { ToolProvider } from '../../../shared/annotator/tool-provider.component';
import { isVideoFrame } from '../../../shared/media-item-utils';
import { useMediaPredictions } from '../../annotator/api/use-media-predictions';
import { PredictionsSetupProvider, usePredictionSetup } from '../../annotator/predictions-setup-provider.component';
import {
    SelectedMediaItemProvider,
    useSelectedMediaItem,
} from '../../annotator/selected-media-item-provider.component';
import { AnnotatorProviders } from './annotator-providers.component';
import { AnnotatorContainer } from './annotator.component';
import { useAnnotationsQuery } from './api/use-annotations-query';
import { SIDEBAR_WIDTH } from './constants';
import { SidebarItems } from './sidebar-items/sidebar-items.component';
import { useAnnotatorMediaTransition } from './use-annotator-media-transition.hook';
import { getInitialAnnotations, getInitialPredictions } from './utils';

type MediaPreviewProps = {
    mediaItem: Media;
    close: () => void;
    onSelectedMediaItem: (item: Media) => void;
};

type MediaPreviewContentProps = {
    items: Media[];
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;
    hasNextPage: boolean;
    isFetchingNextPage: boolean;
    fetchNextPage: () => void;
};

type MediaPreviewPanelsProps = {
    mode: AnnotatorMode;
    changeAnnotatorMode: (mode: AnnotatorMode) => void;
    items: Media[];
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;
    hasNextPage: boolean;
    isFetchingNextPage: boolean;
    fetchNextPage: () => void;
};

const MediaPreviewPanels = ({
    mode,
    changeAnnotatorMode,
    items,
    onClose,
    onSelectedMediaItem,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
}: MediaPreviewPanelsProps) => {
    const { mediaItem } = useSelectedMediaItem();
    const handleMediaTransition = useAnnotatorMediaTransition({ onSelectedMediaItem });

    return (
        <>
            <AnnotatorContainer
                mode={mode}
                items={items}
                onClose={onClose}
                changeAnnotatorMode={changeAnnotatorMode}
                onSelectedMediaItem={handleMediaTransition}
            />

            <View gridArea={'aside'}>
                <SidebarItems
                    items={items}
                    mediaItem={mediaItem}
                    hasNextPage={hasNextPage}
                    isFetchingNextPage={isFetchingNextPage}
                    fetchNextPage={fetchNextPage}
                    onSelectedMediaItem={handleMediaTransition}
                />
            </View>
        </>
    );
};

const MediaPreviewContent = ({
    items,
    onSelectedMediaItem,
    onClose,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
}: MediaPreviewContentProps) => {
    const [mode, setMode] = useState<AnnotatorMode>('annotation');
    const { mediaItem } = useSelectedMediaItem();
    const { selectedModelId } = usePredictionSetup();

    const { data: annotationsData } = useAnnotationsQuery(mediaItem);
    const { data: predictionsData } = useMediaPredictions({
        mediaId: mediaItem.id,
        modelId: selectedModelId,
        range: isVideoFrame(mediaItem)
            ? { start_frame: mediaItem.frame_number, end_frame: mediaItem.frame_number, stride: mediaItem.frame_stride }
            : null,
    });

    const isUserReviewed = annotationsData?.user_reviewed ?? false;

    const initialAnnotations = useMemo(() => {
        return getInitialAnnotations(isUserReviewed, annotationsData?.annotations ?? []);
    }, [isUserReviewed, annotationsData?.annotations]);

    const initialPredictions = useMemo(() => {
        return getInitialPredictions(predictionsData?.flatMap((predictionData) => predictionData.prediction));
    }, [predictionsData]);

    return (
        <ToolProvider mode={mode}>
            <AnnotatorProviders
                mediaItem={mediaItem}
                initialAnnotationsDTO={initialAnnotations}
                initialPredictionsDTO={initialPredictions}
                isUserReviewed={isUserReviewed}
                mode={mode}
            >
                <MediaPreviewPanels
                    mode={mode}
                    changeAnnotatorMode={setMode}
                    items={items}
                    onClose={onClose}
                    onSelectedMediaItem={onSelectedMediaItem}
                    hasNextPage={hasNextPage}
                    isFetchingNextPage={isFetchingNextPage}
                    fetchNextPage={fetchNextPage}
                />
            </AnnotatorProviders>
        </ToolProvider>
    );
};

export const MediaPreview = ({ mediaItem, close, onSelectedMediaItem }: MediaPreviewProps) => {
    const { items, hasNextPage, isFetchingNextPage, fetchNextPage } = useGetDatasetMediaItems();

    return (
        <Dialog
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-50)',
                '--spectrum-dialog-padding-x': 'var(--spectrum-global-dimension-size-250)',
                '--spectrum-dialog-padding-y': 'var(--spectrum-global-dimension-size-250)',
            }}
        >
            <Content>
                <Grid
                    gap='size-125'
                    width='100%'
                    height='100%'
                    rows='auto 1fr auto auto'
                    columns={['size-700', 'minmax(0, 1fr)', SIDEBAR_WIDTH]}
                    areas={[
                        'header header aside',
                        'toolbar canvas aside',
                        'toolbar video-toolbar aside',
                        'toolbar bottom aside',
                    ]}
                >
                    <SelectedMediaItemProvider mediaItem={mediaItem}>
                        <PredictionsSetupProvider>
                            <MediaPreviewContent
                                items={items}
                                onClose={close}
                                onSelectedMediaItem={onSelectedMediaItem}
                                hasNextPage={hasNextPage}
                                isFetchingNextPage={isFetchingNextPage}
                                fetchNextPage={fetchNextPage}
                            />
                        </PredictionsSetupProvider>
                    </SelectedMediaItemProvider>
                </Grid>
            </Content>
        </Dialog>
    );
};
