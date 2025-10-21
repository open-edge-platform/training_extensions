// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Content, Dialog, dimensionValue, Divider, Flex, Grid, Heading, Loading, View } from '@geti/ui';
import { DatasetItem } from 'src/constants/shared-types';
import { AnnotationActionsProvider } from 'src/features/annotator/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from 'src/features/annotator/annotation-visibility-provider.component';
import { AnnotatorProvider } from 'src/features/annotator/annotator-provider.component';

import { ZoomProvider } from '../../../components/zoom/zoom.provider';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas';
import { SelectAnnotationProvider } from '../../annotator/select-annotation-provider.component';
import { useGetDatasetItems } from '../gallery/use-get-dataset-items.hook';
import { ToolSelectionBar } from './primary-toolbar/primary-toolbar.component';
import { SecondaryToolbar } from './secondary-toolbar/secondary-toolbar.component';
import { SidebarItems } from './sidebar-items/sidebar-items.component';

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
    const { items, hasNextPage, isFetchingNextPage, fetchNextPage } = useGetDatasetItems();

    return (
        <Dialog UNSAFE_style={{ width: '95vw', height: '95vh' }}>
            <Heading>Preview</Heading>

            <Divider />

            <Content
                UNSAFE_style={{
                    backgroundColor: 'var(--spectrum-global-color-gray-50)',
                }}
            >
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
                    <AnnotationActionsProvider mediaItem={mediaItem}>
                        <ZoomProvider>
                            <Suspense fallback={<CanvasAreaLoading />}>
                                <SelectAnnotationProvider>
                                    <AnnotationVisibilityProvider>
                                        <AnnotatorProvider mediaItem={mediaItem}>
                                            <View gridArea={'toolbar'}>
                                                <ToolSelectionBar />
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
                                                <AnnotatorCanvas mediaItem={mediaItem} />
                                            </View>
                                        </AnnotatorProvider>
                                    </AnnotationVisibilityProvider>
                                </SelectAnnotationProvider>
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
                        </ZoomProvider>
                    </AnnotationActionsProvider>
                </Grid>
            </Content>
        </Dialog>
    );
};
