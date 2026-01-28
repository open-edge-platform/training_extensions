// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useState } from 'react';

import { Content, Dialog, Flex, Grid, Loading, View } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isObject } from 'lodash-es';

import { $api } from '../../../api/client';
import { ZoomProvider } from '../../../components/zoom/zoom.provider';
import type { Media } from '../../../constants/shared-types';
import { useGetDatasetItems } from '../../../hooks/use-get-dataset-items.hook';
import { AnnotationActionsProvider } from '../../../shared/annotator/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../../../shared/annotator/annotation-visibility-provider.component';
import { AnnotatorProvider } from '../../../shared/annotator/annotator-provider.component';
import { SelectAnnotationProvider } from '../../../shared/annotator/select-annotation-provider.component';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas/annotator-canvas';
import { BottomToolbar } from './bottom-toolbar/bottom-toolbar.component';
import { SIDEBAR_WIDTH } from './constants';
import { PrimaryToolbar } from './primary-toolbar/primary-toolbar.component';
import { AnnotatorCanvasSettings } from './primary-toolbar/settings/annotator-canvas-settings.component';
import { CanvasSettingsProvider } from './primary-toolbar/settings/canvas-settings-provider.component';
import { AnnotatorMode } from './secondary-toolbar/annotator-modes/mode';
import { SecondaryToolbar } from './secondary-toolbar/secondary-toolbar.component';
import { SidebarItems } from './sidebar-items/sidebar-items.component';
import { getInitialAnnotations, getInitialPredictions } from './utils';

const isUnannotatedError = (error: unknown): boolean => {
    return (
        isObject(error) && 'detail' in error && /Dataset item has not been annotated yet/i.test(String(error.detail))
    );
};

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
    const projectId = useProjectIdentifier();
    const [mode, setMode] = useState<AnnotatorMode>('annotation');

    const { data: annotationsData } = $api.useSuspenseQuery(
        'get',
        '/api/projects/{project_id}/dataset/items/{dataset_item_id}/annotations',
        {
            params: { path: { project_id: projectId, dataset_item_id: mediaItem.id } },
        },
        {
            retry: (_failureCount, error: unknown) => !isUnannotatedError(error),
        }
    );

    const isUserReviewed = annotationsData?.user_reviewed ?? false;
    const annotationsDTO = annotationsData?.annotations ?? [];

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

                            <View gridArea={'toolbar'}>{mode === 'annotation' && <PrimaryToolbar />}</View>

                            <View gridArea={'bottom'}>
                                <BottomToolbar isUserReviewed={isUserReviewed} mediaItem={mediaItem} />
                            </View>

                            <View gridArea={'canvas'} overflow={'hidden'}>
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
        <Dialog UNSAFE_style={{ backgroundColor: 'var(--spectrum-global-color-gray-50)' }}>
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
