// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Content, Dialog, dimensionValue, Divider, Flex, Grid, Heading, Loading, View } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isObject } from 'lodash-es';

import { $api } from '../../../api/client';
import { ZoomProvider } from '../../../components/zoom/zoom.provider';
import type { DatasetItem } from '../../../constants/shared-types';
import { useGetDatasetItems } from '../../../hooks/use-get-dataset-items.hook';
import { AnnotationActionsProvider } from '../../../shared/annotator/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../../../shared/annotator/annotation-visibility-provider.component';
import { AnnotatorProvider } from '../../../shared/annotator/annotator-provider.component';
import { SelectAnnotationProvider } from '../../../shared/annotator/select-annotation-provider.component';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas/annotator-canvas';
import { PrimaryToolbar } from './primary-toolbar/primary-toolbar.component';
import { AnnotatorCanvasSettings } from './primary-toolbar/settings/annotator-canvas-settings.component';
import { CanvasSettingsProvider } from './primary-toolbar/settings/canvas-settings-provider.component';
import { SecondaryToolbar } from './secondary-toolbar/secondary-toolbar.component';
import { SidebarItems } from './sidebar-items/sidebar-items.component';

const isUnannotatedError = (error: unknown): boolean => {
    return (
        isObject(error) && 'detail' in error && /Dataset item has not been annotated yet/i.test(String(error.detail))
    );
};

type MediaPreviewProps = {
    mediaItem: DatasetItem;
    close: () => void;
    onSelectedMediaItem: (item: DatasetItem) => void;
};

const CanvasAreaLoading = () => (
    <Flex gridArea={'canvas'} alignContent={'center'} justifyContent={'center'}>
        <Loading size='L' mode='inline' />
    </Flex>
);

export const MediaPreview = ({ mediaItem, close, onSelectedMediaItem }: MediaPreviewProps) => {
    const projectId = useProjectIdentifier();

    const { items, hasNextPage, isFetchingNextPage, fetchNextPage } = useGetDatasetItems();

    const { data: annotationsData } = $api.useQuery(
        'get',
        '/api/projects/{project_id}/dataset/items/{dataset_item_id}/annotations',
        {
            params: { path: { project_id: projectId, dataset_item_id: mediaItem.id } },
        },
        {
            retry: (_failureCount, error: unknown) => !isUnannotatedError(error),
        }
    );

    return (
        <Dialog UNSAFE_style={{ width: '95vw', height: '95vh' }}>
            <Heading>Preview</Heading>

            <Divider />

            <Content UNSAFE_style={{ backgroundColor: 'var(--spectrum-global-color-gray-50)' }}>
                <Grid
                    gap='size-125'
                    width='100%'
                    height='100%'
                    rows='auto 1fr auto'
                    columns='auto 1fr 140px'
                    UNSAFE_style={{
                        // Matches grid gap (size-125) to align with the leftmost element
                        paddingLeft: dimensionValue('size-125'),
                    }}
                    areas={['toolbar header aside', 'toolbar canvas aside', 'toolbar footer aside']}
                >
                    <AnnotationActionsProvider
                        mediaItem={mediaItem}
                        initialAnnotationsDTO={annotationsData?.annotations ?? []}
                        isUserReviewed={annotationsData?.user_reviewed ?? false}
                    >
                        <ZoomProvider>
                            <Suspense fallback={<CanvasAreaLoading />}>
                                <SelectAnnotationProvider>
                                    <AnnotationVisibilityProvider>
                                        <AnnotatorProvider mediaItem={mediaItem}>
                                            <CanvasSettingsProvider>
                                                <View gridArea={'toolbar'}>
                                                    <PrimaryToolbar />
                                                </View>

                                                <View gridArea={'header'}>
                                                    <SecondaryToolbar
                                                        items={items}
                                                        onClose={close}
                                                        mediaItem={mediaItem}
                                                        onSelectedMediaItem={onSelectedMediaItem}
                                                    />
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
                    </AnnotationActionsProvider>
                </Grid>
            </Content>
        </Dialog>
    );
};
