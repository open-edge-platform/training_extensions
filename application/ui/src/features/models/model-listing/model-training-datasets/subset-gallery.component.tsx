// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useState } from 'react';

import { Content, Dialog, DialogContainer, Flex, Grid, Loading, Size, Text, View, ViewModes } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { GridLayoutOptions } from 'react-aria-components';

import { MediaItem } from '../../../../components/media-item/media-item.component';
import { MediaThumbnail } from '../../../../components/media-thumbnail/media-thumbnail.component';
import { VirtualizerGridLayout } from '../../../../components/virtualizer-grid-layout/virtualizer-grid-layout.component';
import type { DatasetRevisionItem } from '../../../../constants/shared-types';
import {
    SelectedMediaItemProvider,
    useSelectedMediaItem,
} from '../../../../features/annotator/selected-media-item-provider.component';
import { useAnnotationsQuery } from '../../../../features/dataset/media-preview/api/use-annotations-query';
import { ReadOnlyAnnotatorProviders } from '../../../../features/dataset/media-preview/read-only-annotator-providers.component';
import { ReadOnlyAnnotator } from '../../../../features/dataset/media-preview/read-only-annotator.component';
import { getInitialAnnotations } from '../../../../features/dataset/media-preview/utils';
import { type GalleryViewMode } from '../../../../shared/gallery-view-modes';
import { getDatasetRevisionThumbnailUrl } from '../../../../shared/media-url.utils';
import { datasetRevisionItemToMedia } from './utils';

const VIEW_MODE_SETTINGS: Record<GalleryViewMode, GridLayoutOptions> = {
    [ViewModes.LARGE]: { minItemSize: new Size(180, 180), minSpace: new Size(6, 6), preserveAspectRatio: true },
    [ViewModes.MEDIUM]: { minItemSize: new Size(120, 120), minSpace: new Size(4, 4), preserveAspectRatio: true },
    [ViewModes.SMALL]: { minItemSize: new Size(80, 80), minSpace: new Size(4, 4), preserveAspectRatio: true },
};

type SubsetGalleryProps = {
    items: DatasetRevisionItem[];
    datasetRevisionId: string;
    viewMode: GalleryViewMode;
    fetchNextPage: () => void;
    hasNextPage: boolean;
    isFetchingNextPage: boolean;
    isPending: boolean;
};

type SubsetMediaDialogProps = {
    item: DatasetRevisionItem;
    onClose: () => void;
};

type SubsetMediaDialogContentProps = {
    item: DatasetRevisionItem;
    onClose: () => void;
};

const SubsetMediaDialogContent = ({ item, onClose }: SubsetMediaDialogContentProps) => {
    const { mediaItem, image } = useSelectedMediaItem();
    const { data: annotationsData } = useAnnotationsQuery(mediaItem);

    const annotationsDTO = annotationsData?.annotations ?? [];
    const isUserReviewed = annotationsData?.user_reviewed ?? false;

    return (
        <ReadOnlyAnnotatorProviders
            key={mediaItem.id}
            mediaItem={mediaItem}
            initialAnnotationsDTO={getInitialAnnotations(isUserReviewed, annotationsDTO)}
            isUserReviewed={isUserReviewed}
        >
            <ReadOnlyAnnotator
                image={image}
                mediaItem={mediaItem}
                onClose={onClose}
                subset={item.subset}
                hasAnnotationStatus={false}
            />
        </ReadOnlyAnnotatorProviders>
    );
};

const SubsetMediaDialog = ({ item, onClose }: SubsetMediaDialogProps) => {
    const mediaItem = datasetRevisionItemToMedia(item);

    return (
        <Dialog>
            <Content>
                <Grid
                    gap='size-125'
                    width='100%'
                    height='100%'
                    rows='auto 1fr auto'
                    columns={['1fr']}
                    areas={['header', 'canvas', 'bottom']}
                >
                    <SelectedMediaItemProvider mediaItem={mediaItem}>
                        <SubsetMediaDialogContent item={item} onClose={onClose} />
                    </SelectedMediaItemProvider>
                </Grid>
            </Content>
        </Dialog>
    );
};

export const SubsetGallery = ({
    items,
    viewMode,
    datasetRevisionId,
    hasNextPage,
    isFetchingNextPage,
    isPending,
    fetchNextPage,
}: SubsetGalleryProps) => {
    const projectId = useProjectIdentifier();
    const [selectedItem, setSelectedItem] = useState<DatasetRevisionItem | null>(null);

    if (isPending) {
        return (
            <Flex height={'100%'} alignItems={'center'} justifyContent={'center'}>
                <Loading mode='inline' />
            </Flex>
        );
    }

    if (items.length === 0) {
        return (
            <Flex height={'100%'} alignItems={'center'} justifyContent={'center'}>
                <Text>No items in this subset</Text>
            </Flex>
        );
    }

    return (
        <>
            <View height={'size-5000'} width={'100%'}>
                <VirtualizerGridLayout
                    items={items}
                    ariaLabel={'subset media grid'}
                    selectionMode='none'
                    layoutOptions={VIEW_MODE_SETTINGS[viewMode]}
                    isLoadingMore={isFetchingNextPage}
                    onLoadMore={() => hasNextPage && fetchNextPage()}
                    contentItem={(item) => (
                        <MediaItem
                            contentElement={() => (
                                <MediaThumbnail
                                    // TODO: Revisit this once API supports required props in DatasetRevisionItem
                                    item={{ ...item, type: 'image' }}
                                    alt={`${item.subset} item`}
                                    url={getDatasetRevisionThumbnailUrl(projectId, datasetRevisionId, item.id)}
                                    onDoubleClick={() => setSelectedItem(item)}
                                />
                            )}
                        />
                    )}
                />
            </View>

            <DialogContainer type={'fullscreen'} onDismiss={() => setSelectedItem(null)}>
                {selectedItem && (
                    <Suspense fallback={<Loading />}>
                        <SubsetMediaDialog item={selectedItem} onClose={() => setSelectedItem(null)} />
                    </Suspense>
                )}
            </DialogContainer>
        </>
    );
};
