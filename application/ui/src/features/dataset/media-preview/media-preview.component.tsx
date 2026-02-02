// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useState } from 'react';

import { Content, Dialog, Flex, Grid, Loading, View } from '@geti/ui';
import { clsx } from 'clsx';

import { ZoomProvider } from '../../../components/zoom/zoom.provider';
import type { Media } from '../../../constants/shared-types';
import { useGetDatasetItems } from '../../../hooks/use-get-dataset-items.hook';
import { AnnotationActionsProvider } from '../../../shared/annotator/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../../../shared/annotator/annotation-visibility-provider.component';
import { AnnotatorProvider } from '../../../shared/annotator/annotator-provider.component';
import { SelectAnnotationProvider } from '../../../shared/annotator/select-annotation-provider.component';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas/annotator-canvas';
import { useAnnotationsQuery } from './api/use-annotations-query';
import { BottomToolbar } from './bottom-toolbar/bottom-toolbar.component';
import { SIDEBAR_WIDTH } from './constants';
import { PrimaryToolbar } from './primary-toolbar/primary-toolbar.component';
import { AnnotatorCanvasSettings } from './primary-toolbar/settings/annotator-canvas-settings.component';
import { CanvasSettingsProvider } from './primary-toolbar/settings/canvas-settings-provider.component';
import { AnnotatorMode } from './secondary-toolbar/annotator-modes/mode';
import { SecondaryToolbar } from './secondary-toolbar/secondary-toolbar.component';
import { SidebarItems } from './sidebar-items/sidebar-items.component';
import { getInitialAnnotations, getInitialPredictions } from './utils';

import styles from './media-preview.module.scss';

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

    const isReadOnlyCanvas = mode === 'prediction';

    return (
        <AnnotationActionsProvider
            key={mediaItem.id}
            mediaItem={mediaItem}
            initialAnnotationsDTO={getInitialAnnotations(mode, isUserReviewed, annotationsDTO)}
            initialPredictionsDTO={getInitialPredictions(mode, isUserReviewed, annotationsDTO)}
            isUserReviewed={isUserReviewed}
            mode={mode}
        >
            <SelectAnnotationProvider>
                <AnnotationVisibilityProvider>
                    <AnnotatorProvider mediaItem={mediaItem}>
                        <CanvasSettingsProvider>
                            <View gridArea={'header'}>
                                <SecondaryToolbar
                                    items={items}
                                    onClose={onClose}
                                    mediaItem={mediaItem}
                                    onSelectedMediaItem={onSelectedMediaItem}
                                    mode={mode}
                                    onModeChange={setMode}
                                />
                            </View>

                            <div
                                style={{ gridArea: 'toolbar' }}
                                aria-label={'primary toolbar'}
                                aria-disabled={isReadOnlyCanvas}
                                className={clsx({ [styles.primaryToolbarDisabled]: isReadOnlyCanvas })}
                            >
                                <PrimaryToolbar />
                            </div>

                            <View gridArea={'bottom'}>
                                <BottomToolbar isUserReviewed={isUserReviewed} mediaItem={mediaItem} />
                            </View>

                            <View
                                gridArea={'canvas'}
                                overflow={'hidden'}
                                UNSAFE_className={clsx({ [styles.readOnlyCanvas]: isReadOnlyCanvas })}
                            >
                                <AnnotatorCanvasSettings>
                                    <AnnotatorCanvas mediaItem={mediaItem} />
                                </AnnotatorCanvasSettings>
                            </View>
                        </CanvasSettingsProvider>
                    </AnnotatorProvider>
                </AnnotationVisibilityProvider>
            </SelectAnnotationProvider>
        </AnnotationActionsProvider>
    );
};

export const MediaPreview = ({ mediaItem, close, onSelectedMediaItem }: MediaPreviewProps) => {
    const { items, hasNextPage, isFetchingNextPage, fetchNextPage } = useGetDatasetItems();

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
                    columns={['size-800', '1fr', SIDEBAR_WIDTH]}
                    areas={['header header aside', 'toolbar canvas aside', 'toolbar bottom aside']}
                >
                    <ZoomProvider>
                        <Suspense fallback={<CanvasAreaLoading />}>
                            <MediaPreviewContent
                                items={items}
                                mediaItem={mediaItem}
                                onClose={close}
                                onSelectedMediaItem={onSelectedMediaItem}
                            />
                        </Suspense>
                    </ZoomProvider>

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
