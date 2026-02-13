// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useState } from 'react';

import { Content, Dialog, Flex, Grid, Loading, View } from '@geti/ui';
import { QueryClient, useQueryClient } from '@tanstack/react-query';
import { useGetDatasetMediaItems } from 'hooks/use-get-dataset-media-items.hook';

import type { Media } from '../../../constants/shared-types';
import { ToolProvider } from '../../../shared/annotator/tool-provider.component';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas/annotator-canvas';
import { VideoPlayerProvider } from '../../annotator/video-player/video-player-provider.component';
import { useSelectedData } from '../selected-data-provider.component';
import { AnnotatorProviders } from './annotator-providers.component';
import { useAnnotationsQuery } from './api/use-annotations-query';
import { BottomToolbar } from './bottom-toolbar/bottom-toolbar.component';
import { SIDEBAR_WIDTH } from './constants';
import { PrimaryToolbar } from './primary-toolbar/primary-toolbar.component';
import { AnnotatorCanvasSettings } from './primary-toolbar/settings/annotator-canvas-settings.component';
import { ReadOnlyAnnotator } from './read-only-annotator.component';
import { AnnotatorMode } from './secondary-toolbar/annotator-modes/mode';
import { SecondaryToolbar } from './secondary-toolbar/secondary-toolbar.component';
import { getNextItem } from './secondary-toolbar/util';
import { SidebarItems } from './sidebar-items/sidebar-items.component';
import { getInitialAnnotations, getInitialPredictions } from './utils';

type MediaPreviewProps = {
    mediaItem: Media;
    close: () => void;
    onSelectedMediaItem: (item: Media) => void;
};

type MediaPreviewContentProps = {
    items: Media[];
    mediaItem: Media;
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;
};

const CanvasAreaLoading = () => (
    <Flex gridArea={'canvas'} alignContent={'center'} justifyContent={'center'}>
        <Loading size='L' mode='inline' />
    </Flex>
);

const invalidateMediaItemAnnotations = (queryClient: QueryClient) => {
    queryClient.invalidateQueries({
        queryKey: ['get', '/api/projects/{project_id}/dataset/items/{dataset_item_id}/annotations'],
    });
};

const MediaPreviewContent = ({ items, mediaItem, onSelectedMediaItem, onClose }: MediaPreviewContentProps) => {
    const [mode, setMode] = useState<AnnotatorMode>('annotation');

    const { data: annotationsData } = useAnnotationsQuery(mediaItem.id);

    const isUserReviewed = annotationsData?.user_reviewed ?? false;
    const annotationsDTO = annotationsData?.annotations ?? [];
    const queryClient = useQueryClient();
    const { setMediaState } = useSelectedData();

    const selectedIndex = items.findIndex((item) => item.id === mediaItem.id);

    const handleSubmitAnnotations = async () => {
        setMediaState((prev) => {
            const newState = new Map(prev);

            newState.set(String(mediaItem.id), 'accepted');

            return newState;
        });

        const nextItem = getNextItem(items.length - 1, selectedIndex);
        onSelectedMediaItem(items[nextItem]);

        const isLastItem = selectedIndex === items.length - 1;
        isLastItem && invalidateMediaItemAnnotations(queryClient);
    };

    return (
        <ToolProvider mode={mode}>
            <AnnotatorProviders
                key={mediaItem.id}
                mediaItem={mediaItem}
                initialAnnotationsDTO={getInitialAnnotations(mode, isUserReviewed, annotationsDTO)}
                initialPredictionsDTO={getInitialPredictions(mode, isUserReviewed, annotationsDTO)}
                isUserReviewed={isUserReviewed}
                mode={mode}
            >
                <VideoPlayerProvider>
                    {mode === 'prediction' ? (
                        <ReadOnlyAnnotator
                            mediaItem={mediaItem}
                            isUserReviewed={isUserReviewed}
                            onModeChange={setMode}
                            onClose={onClose}
                            onAcceptPrediction={handleSubmitAnnotations}
                        />
                    ) : (
                        <>
                            <View gridArea={'header'}>
                                <SecondaryToolbar
                                    mode={mode}
                                    items={items}
                                    onClose={onClose}
                                    mediaItem={mediaItem}
                                    onSelectedMediaItem={onSelectedMediaItem}
                                    onModeChange={setMode}
                                    onAcceptPrediction={handleSubmitAnnotations}
                                />
                            </View>

                            <View gridArea={'toolbar'} aria-label={'primary toolbar'}>
                                <PrimaryToolbar />
                            </View>

                            <View gridArea={'bottom'}>
                                <BottomToolbar isUserReviewed={isUserReviewed} mediaItem={mediaItem} />
                            </View>

                            <View gridArea={'canvas'} overflow={'hidden'}>
                                <AnnotatorCanvasSettings>
                                    <AnnotatorCanvas mediaItem={mediaItem} />
                                </AnnotatorCanvasSettings>
                            </View>
                        </>
                    )}
                </VideoPlayerProvider>
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
                    rows='auto 1fr auto'
                    columns={['size-700', '1fr', SIDEBAR_WIDTH]}
                    areas={['header header aside', 'toolbar canvas aside', 'toolbar bottom aside']}
                >
                    <Suspense fallback={<CanvasAreaLoading />}>
                        <MediaPreviewContent
                            items={items}
                            mediaItem={mediaItem}
                            onClose={close}
                            onSelectedMediaItem={onSelectedMediaItem}
                        />
                    </Suspense>

                    <View gridArea={'aside'}>
                        <SidebarItems
                            items={items}
                            mediaItem={mediaItem}
                            hasNextPage={hasNextPage}
                            isFetchingNextPage={isFetchingNextPage}
                            fetchNextPage={fetchNextPage}
                            onSelectedMediaItem={onSelectedMediaItem}
                        />
                    </View>
                </Grid>
            </Content>
        </Dialog>
    );
};
