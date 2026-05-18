// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Checkbox, DialogContainer, dimensionValue, Flex, Size, ViewModes } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty } from 'lodash-es';
import { GridLayoutOptions } from 'react-aria-components';

import { MediaItem } from '../../../components/media-item/media-item.component';
import { MediaThumbnail } from '../../../components/media-thumbnail/media-thumbnail.component';
import { VirtualizerGridLayout } from '../../../components/virtualizer-grid-layout/virtualizer-grid-layout.component';
import type { Media } from '../../../constants/shared-types';
import { type GalleryViewMode } from '../../../shared/gallery-view-modes';
import { getMediaDownloadUrl, getThumbnailUrl } from '../../../shared/media-url.utils';
import { MediaPreview } from '../media-preview/media-preview.component';
import { useSelectedData } from '../providers/selected-data-provider.component';
import { AnnotationStatusIcon } from './annotation-state-icon.component';
import { BulkLabelsAssignmentDialog } from './bulk-labels-assignment/bulk-labels-assignment-dialog.component';
import { DatasetDropZone } from './drop-zone.component';
import { EmptyDataset } from './empty-dataset.component';
import { useSelectDatasetItem } from './hooks/use-select-dataset-item.hook';
import { MediaItemActions } from './media-item-actions/media-item-actions.component';
import { MediaItemContextualHelp } from './media-item-contextual-help/media-item-contextual-help.component';
import { useUploadFiles } from './use-upload-files';

type GalleryProps = {
    items: Media[];
    viewMode: GalleryViewMode;
    isPending: boolean;
    hasActiveFilter: boolean;
    isFetchingNextPage: boolean;
    fetchNextPage: () => void;
    isMediaItemReviewedById: (mediaItemId: string) => boolean;
};

const VIEW_MODE_SETTINGS: Record<GalleryViewMode, GridLayoutOptions> = {
    [ViewModes.LARGE]: { minItemSize: new Size(300, 300), minSpace: new Size(10, 10), preserveAspectRatio: true },
    [ViewModes.MEDIUM]: { minItemSize: new Size(200, 200), minSpace: new Size(6, 6), preserveAspectRatio: true },
    [ViewModes.SMALL]: { minItemSize: new Size(120, 120), minSpace: new Size(4, 4), preserveAspectRatio: true },
};

type GalleryListProps = {
    items: Media[];
    viewMode: GalleryViewMode;
    isFetchingNextPage: boolean;
    fetchNextPage: () => void;
    isMediaItemReviewedById: (mediaItemId: string) => boolean;
    onSelectedMediaItemChange: (item: Media) => void;
};

const GalleryList = ({
    items,
    viewMode,
    isFetchingNextPage,
    fetchNextPage,
    onSelectedMediaItemChange,
    isMediaItemReviewedById,
}: GalleryListProps) => {
    const projectId = useProjectIdentifier();
    const { selectedKeys, setSelectedKeys, toggleSelectedKeys } = useSelectedData();

    const isSetSelectedKeys = selectedKeys instanceof Set;

    return (
        <VirtualizerGridLayout
            items={items}
            ariaLabel='data-collection-grid'
            selectionMode='multiple'
            selectedKeys={selectedKeys}
            layoutOptions={VIEW_MODE_SETTINGS[viewMode]}
            isLoadingMore={isFetchingNextPage}
            onLoadMore={fetchNextPage}
            onSelectionChange={setSelectedKeys}
            contentItem={(item) => {
                const mediaUrl = getThumbnailUrl(projectId, item.id);
                const downloadUrl = getMediaDownloadUrl(projectId, item.id);
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
                                UNSAFE_style={{ margin: dimensionValue('size-150') }}
                            >
                                <Checkbox
                                    aria-label={`Select media item ${item.name}`}
                                    onChange={() => toggleSelectedKeys([String(item.id)])}
                                    isSelected={isSetSelectedKeys && selectedKeys.has(String(item.id))}
                                />
                            </Flex>
                        )}
                        topRightElement={() => (
                            <Flex alignItems={'center'} gap={'size-50'}>
                                <MediaItemContextualHelp item={item} />

                                <MediaItemActions
                                    id={item.id}
                                    onDeleted={toggleSelectedKeys}
                                    mediaUrl={downloadUrl}
                                    mediaFileName={mediaFileName}
                                    onAnnotate={() => onSelectedMediaItemChange(item)}
                                />
                            </Flex>
                        )}
                        bottomRightElement={() => (
                            <AnnotationStatusIcon state={isMediaItemReviewedById(item.id) ? 'accepted' : undefined} />
                        )}
                    />
                );
            }}
        />
    );
};

export const Gallery = ({
    items,
    viewMode,
    isPending,
    hasActiveFilter,
    isFetchingNextPage,
    fetchNextPage,
    isMediaItemReviewedById,
}: GalleryProps) => {
    const { selectedMediaItem, onSelectedMediaItemChange } = useSelectDatasetItem();

    const { isClassification, uploadFiles, clearFilesForLabelAssignment, filesForLabelAssignment } = useUploadFiles();

    const content =
        !isPending && isEmpty(items) ? (
            <EmptyDataset hasActiveFilter={hasActiveFilter} />
        ) : (
            <GalleryList
                items={items}
                viewMode={viewMode}
                fetchNextPage={fetchNextPage}
                isMediaItemReviewedById={isMediaItemReviewedById}
                onSelectedMediaItemChange={onSelectedMediaItemChange}
                isFetchingNextPage={isFetchingNextPage}
            />
        );

    return (
        <>
            <DatasetDropZone onFilesDropped={uploadFiles}>
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

            {isClassification && (
                <BulkLabelsAssignmentDialog files={filesForLabelAssignment} onClose={clearFilesForLabelAssignment} />
            )}
        </>
    );
};
