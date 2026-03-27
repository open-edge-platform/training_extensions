// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Checkbox, DialogContainer, Flex, Size, ViewModes } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty } from 'lodash-es';
import { GridLayoutOptions } from 'react-aria-components';

import { MediaItem } from '../../../components/media-item/media-item.component';
import { MediaThumbnail } from '../../../components/media-thumbnail/media-thumbnail.component';
import { VirtualizerGridLayout } from '../../../components/virtualizer-grid-layout/virtualizer-grid-layout.component';
import type { DatasetItemAnnotationStatus, Media, MediaDTO } from '../../../constants/shared-types';
import { useGetDatasetItemsById } from '../../../hooks/use-get-dataset-items-by-id.hook';
import { getMediaBinaryUrl, getThumbnailUrl } from '../../../shared/media-url.utils';
import { MediaPreview } from '../media-preview/media-preview.component';
import { useSelectedData } from '../providers/selected-data-provider.component';
import { AnnotationStatusIcon } from './annotation-state-icon.component';
import { DatasetDropZone } from './drop-zone.component';
import { EmptyDataset } from './empty-dataset.component';
import { useSelectDatasetItem } from './hooks/use-select-dataset-item.hook';
import { MediaItemActions } from './media-item-actions/media-item-actions.component';

type GalleryProps = {
    items: Media[];
    annotationStatus?: DatasetItemAnnotationStatus;
    viewMode: ViewModes;
    isPending: boolean;
    hasActiveFilter: boolean;
    hasNextPage: boolean;
    isFetchingNextPage: boolean;
    fetchNextPage: () => void;
    onFilesDropped?: (files: File[]) => void | Promise<MediaDTO[]>;
};

// DetailsView isn’t needed, so we’re forcing the cast to prevent TS from complaining about missing properties
const VIEW_MODE_SETTINGS = {
    [ViewModes.LARGE]: { minItemSize: new Size(300, 300), minSpace: new Size(10, 10), preserveAspectRatio: true },
    [ViewModes.MEDIUM]: { minItemSize: new Size(200, 200), minSpace: new Size(6, 6), preserveAspectRatio: true },
    [ViewModes.SMALL]: { minItemSize: new Size(120, 120), minSpace: new Size(4, 4), preserveAspectRatio: true },
} as Record<ViewModes, GridLayoutOptions>;

export const Gallery = ({
    items,
    annotationStatus,
    viewMode,
    isPending,
    hasActiveFilter,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    onFilesDropped,
}: GalleryProps) => {
    const projectId = useProjectIdentifier();
    const { selectedMediaItem, onSelectedMediaItemChange } = useSelectDatasetItem();
    const { selectedKeys, setSelectedKeys, toggleSelectedKeys } = useSelectedData();
    const { datasetItemsById } = useGetDatasetItemsById({ limit: items.length, annotationStatus });

    const isSetSelectedKeys = selectedKeys instanceof Set;

    const content =
        !isPending && isEmpty(items) ? (
            <EmptyDataset hasActiveFilter={hasActiveFilter} />
        ) : (
            <VirtualizerGridLayout
                items={items}
                ariaLabel='data-collection-grid'
                selectionMode='multiple'
                selectedKeys={selectedKeys}
                layoutOptions={VIEW_MODE_SETTINGS[viewMode]}
                isLoadingMore={isFetchingNextPage}
                onLoadMore={() => hasNextPage && fetchNextPage()}
                onSelectionChange={setSelectedKeys}
                contentItem={(item) => {
                    const mediaUrl = getThumbnailUrl(projectId, item.id);
                    const fullMediaUrl = getMediaBinaryUrl(projectId, item.id);
                    const mediaFileName = `${item.name}.${item.format}`;

                    return (
                        <MediaItem
                            contentElement={() => (
                                <MediaThumbnail
                                    item={item}
                                    alt={item.name}
                                    url={mediaUrl}
                                    onDoubleClick={() => onSelectedMediaItemChange(item)}
                                />
                            )}
                            topLeftElement={() => (
                                <Flex
                                    width={'size-200'}
                                    height={'size-200'}
                                    alignItems={'center'}
                                    justifyContent={'center'}
                                >
                                    <Checkbox
                                        aria-label={`Select media item ${item.name}`}
                                        onChange={() => toggleSelectedKeys([String(item.id)])}
                                        isSelected={isSetSelectedKeys && selectedKeys.has(String(item.id))}
                                    />
                                </Flex>
                            )}
                            topRightElement={() => (
                                <MediaItemActions
                                    id={item.id}
                                    onDeleted={toggleSelectedKeys}
                                    mediaUrl={fullMediaUrl}
                                    mediaFileName={mediaFileName}
                                    onAnnotate={() => onSelectedMediaItemChange(item)}
                                />
                            )}
                            bottomRightElement={() => {
                                const mediaItemId = String(item.id);
                                const isUserReviewed = datasetItemsById.get(mediaItemId) ?? false;

                                return <AnnotationStatusIcon state={isUserReviewed ? 'accepted' : undefined} />;
                            }}
                        />
                    );
                }}
            />
        );

    return (
        <DatasetDropZone onFilesDropped={onFilesDropped}>
            {content}

            <DialogContainer type={'fullscreenTakeover'} onDismiss={() => onSelectedMediaItemChange(null)}>
                {selectedMediaItem !== null && (
                    <MediaPreview
                        mediaItem={selectedMediaItem}
                        close={() => onSelectedMediaItemChange(null)}
                        onSelectedMediaItem={onSelectedMediaItemChange}
                    />
                )}
            </DialogContainer>
        </DatasetDropZone>
    );
};
