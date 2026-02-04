// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useState } from 'react';

import { Content, Dialog, Flex, Grid, Loading, View } from '@geti/ui';
import { useGetDatasetMediaItems } from 'hooks/use-get-dataset-media-items.hook';

import type { Media } from '../../../constants/shared-types';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas/annotator-canvas';
import { AnnotatorProviders } from './annotator-providers.component';
import { useAnnotationsQuery } from './api/use-annotations-query';
import { BottomToolbar } from './bottom-toolbar/bottom-toolbar.component';
import { SIDEBAR_WIDTH } from './constants';
import { PrimaryToolbar } from './primary-toolbar/primary-toolbar.component';
import { AnnotatorCanvasSettings } from './primary-toolbar/settings/annotator-canvas-settings.component';
import { ReadOnlyAnnotator } from './read-only-annotator.component';
import { AnnotatorMode } from './secondary-toolbar/annotator-modes/mode';
import { SecondaryToolbar } from './secondary-toolbar/secondary-toolbar.component';
import { SidebarItems } from './sidebar-items/sidebar-items.component';
import { getInitialAnnotations, getInitialPredictions } from './utils';

import classes from './media-preview.module.scss';

type MediaPreviewProps = {
    mediaItem: Media;
    close: () => void;
    onSelectedMediaItem: (item: Media) => void;
};

const CanvasAreaLoading = () => (
    <Flex gridArea={'canvas'} alignContent={'center'} justifyContent={'center'}>
        <Loading size='L' mode='inline' />
    </Flex>
);

type MediaPreviewContentProps = {
    items: Media[];
    mediaItem: Media;
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;
};

const MediaPreviewContent = ({ items, mediaItem, onSelectedMediaItem, onClose }: MediaPreviewContentProps) => {
    const [mode, setMode] = useState<AnnotatorMode>('annotation');

    const { data: annotationsData } = useAnnotationsQuery(mediaItem.id);

    const isUserReviewed = annotationsData?.user_reviewed ?? false;
    const annotationsDTO = annotationsData?.annotations ?? [];

    return (
        <AnnotatorProviders
            key={mediaItem.id}
            mediaItem={mediaItem}
            initialAnnotationsDTO={getInitialAnnotations(mode, isUserReviewed, annotationsDTO)}
            initialPredictionsDTO={getInitialPredictions(mode, isUserReviewed, annotationsDTO)}
            isUserReviewed={isUserReviewed}
            mode={mode}
        >
            {mode === 'prediction' ? (
                <ReadOnlyAnnotator
                    mediaItem={mediaItem}
                    isUserReviewed={isUserReviewed}
                    onModeChange={setMode}
                    onClose={onClose}
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
                        />
                    </View>

                    <View gridArea={'toolbar'} aria-label={'primary toolbar'} UNSAFE_className={classes.primaryToolbar}>
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
        </AnnotatorProviders>
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
