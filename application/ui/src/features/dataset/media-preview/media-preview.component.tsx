// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from 'react';

import { ActionButton, Content, Dialog, Flex, Grid, View } from '@geti/ui';
import { Close, Expand, FitScreen } from '@geti/ui/icons';
import { useDatasetMediaWithReviewStatus } from 'hooks/use-dataset-media-with-review-status.hook';

import type { DatasetSubset, Media } from '../../../constants/shared-types';
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
import { getInitialAnnotations, useAnnotatorMode } from './utils';

// On small screens (tablets / small laptops) the annotator always opens fullscreen without window controls.
const TABLET_MEDIA_QUERY = '(max-width: 1024px)';

const useMatchMedia = (query: string): boolean => {
    const [matches, setMatches] = useState(() =>
        typeof window !== 'undefined' ? window.matchMedia(query).matches : false
    );

    useEffect(() => {
        const mediaQueryList = window.matchMedia(query);
        const handleChange = (event: MediaQueryListEvent) => setMatches(event.matches);

        setMatches(mediaQueryList.matches);
        mediaQueryList.addEventListener('change', handleChange);

        return () => mediaQueryList.removeEventListener('change', handleChange);
    }, [query]);

    return matches;
};

type MediaPreviewProps = {
    mediaItem: Media;
    close: () => void;
    onSelectedMediaItem: (item: Media) => void;
};

type MediaPreviewContentProps = {
    items: Media[];
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;
    isFetchingNextPage: boolean;
    fetchNextPage: () => void;
    isMediaItemReviewedById: (mediaItemId: string) => boolean;
};

type MediaPreviewPanelsProps = {
    mode: AnnotatorMode;
    changeAnnotatorMode: (mode: AnnotatorMode) => void;
    items: Media[];
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;
    isFetchingNextPage: boolean;
    fetchNextPage: () => void;
    isMediaItemReviewedById: (mediaItemId: string) => boolean;
    isCurrentMediaReviewed: boolean;
    subset: DatasetSubset;
};

const MediaPreviewPanels = ({
    mode,
    subset,
    changeAnnotatorMode,
    items,
    onClose,
    onSelectedMediaItem,
    isFetchingNextPage,
    fetchNextPage,
    isMediaItemReviewedById,
    isCurrentMediaReviewed,
}: MediaPreviewPanelsProps) => {
    const { mediaItem } = useSelectedMediaItem();
    const handleMediaTransition = useAnnotatorMediaTransition({ onSelectedMediaItem });

    return (
        <>
            <AnnotatorContainer
                mode={mode}
                items={items}
                subset={subset}
                onClose={onClose}
                isUserReviewed={isCurrentMediaReviewed}
                changeAnnotatorMode={changeAnnotatorMode}
                onSelectedMediaItem={handleMediaTransition}
            />

            <View gridArea={'aside'}>
                <SidebarItems
                    items={items}
                    mediaItem={mediaItem}
                    isFetchingNextPage={isFetchingNextPage}
                    fetchNextPage={fetchNextPage}
                    isUserReviewed={isMediaItemReviewedById}
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
    isFetchingNextPage,
    fetchNextPage,
    isMediaItemReviewedById,
}: MediaPreviewContentProps) => {
    const { mediaItem } = useSelectedMediaItem();
    const { selectedModel, selectedDevice } = usePredictionSetup();

    const { data: annotationsData } = useAnnotationsQuery(mediaItem);
    const { data: predictionsData } = useMediaPredictions({
        mediaId: mediaItem.id,
        selectedModel,
        device: selectedDevice,
        range: isVideoFrame(mediaItem)
            ? { start_frame: mediaItem.frame_number, end_frame: mediaItem.frame_number, stride: mediaItem.frame_stride }
            : null,
    });

    const isCurrentMediaReviewed = annotationsData?.user_reviewed ?? false;
    const subset: DatasetSubset = annotationsData?.subset ?? 'unassigned';

    const initialAnnotations = useMemo(() => {
        return getInitialAnnotations(isCurrentMediaReviewed, annotationsData?.annotations ?? []);
    }, [isCurrentMediaReviewed, annotationsData?.annotations]);

    const initialPredictions = useMemo(() => {
        return predictionsData?.flatMap((predictionData) => predictionData.prediction) ?? [];
    }, [predictionsData]);

    const [mode, setMode] = useAnnotatorMode();

    return (
        <ToolProvider>
            <AnnotatorProviders
                mediaItem={mediaItem}
                initialAnnotationsDTO={initialAnnotations}
                initialPredictionsDTO={initialPredictions}
                isUserReviewed={isCurrentMediaReviewed}
                mode={mode}
            >
                <MediaPreviewPanels
                    mode={mode}
                    changeAnnotatorMode={setMode}
                    items={items}
                    onClose={onClose}
                    onSelectedMediaItem={onSelectedMediaItem}
                    isFetchingNextPage={isFetchingNextPage}
                    fetchNextPage={fetchNextPage}
                    isMediaItemReviewedById={isMediaItemReviewedById}
                    isCurrentMediaReviewed={isCurrentMediaReviewed}
                    subset={subset}
                />
            </AnnotatorProviders>
        </ToolProvider>
    );
};

export const MediaPreview = ({ mediaItem, close, onSelectedMediaItem }: MediaPreviewProps) => {
    const { items, isFetchingNextPage, fetchNextPage, isMediaItemReviewedById } = useDatasetMediaWithReviewStatus();

    const [isFullscreen, setIsFullscreen] = useState(false);
    const isTablet = useMatchMedia(TABLET_MEDIA_QUERY);
    const isFullScreenSized = isTablet || isFullscreen;

    return (
        <Dialog
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-50)',
                '--spectrum-dialog-padding-x': 'var(--spectrum-global-dimension-size-250)',
                '--spectrum-dialog-padding-y': 'var(--spectrum-global-dimension-size-250)',
                '--spectrum-dialog-max-width': 'none',
                position: 'relative',
                width: isFullScreenSized ? '100vw' : '90vw',
                height: isFullScreenSized ? '100vh' : '90vh',
                maxWidth: isFullScreenSized ? '100vw' : '90vw',
                maxHeight: isFullScreenSized ? '100vh' : '90vh',
                borderRadius: isFullScreenSized ? 0 : undefined,
            }}
        >
            {!isTablet && (
                <Flex position='absolute' top='size-150' right='size-150' gap='size-100' UNSAFE_style={{ zIndex: 10 }}>
                    <ActionButton
                        isQuiet
                        aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                        onPress={() => setIsFullscreen((prev) => !prev)}
                    >
                        {isFullscreen ? <FitScreen /> : <Expand />}
                    </ActionButton>
                    <ActionButton isQuiet aria-label='Close annotator' onPress={close}>
                        <Close />
                    </ActionButton>
                </Flex>
            )}
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
                                isFetchingNextPage={isFetchingNextPage}
                                fetchNextPage={fetchNextPage}
                                isMediaItemReviewedById={isMediaItemReviewedById}
                            />
                        </PredictionsSetupProvider>
                    </SelectedMediaItemProvider>
                </Grid>
            </Content>
        </Dialog>
    );
};
