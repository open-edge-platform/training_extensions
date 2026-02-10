// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Checkbox, DialogContainer, Flex, Size, ViewModes } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { GridLayoutOptions } from 'react-aria-components';

import { MediaItem } from '../../../components/media-item/media-item.component';
import { MediaThumbnail } from '../../../components/media-thumbnail/media-thumbnail.component';
import { VirtualizerGridLayout } from '../../../components/virtualizer-grid-layout/virtualizer-grid-layout.component';
import type { Media } from '../../../constants/shared-types';
import { getThumbnailUrl } from '../../../shared/media-url.utils';
import { MediaPreview } from '../media-preview/media-preview.component';
import { useSelectedData } from '../selected-data-provider.component';
import { DeleteMediaItem } from './delete-media-item/delete-media-item.component';
import { useSelectDatasetItem } from './hooks/use-select-dataset-item.hook';

type GalleryProps = {
    items: Media[];
    viewMode: ViewModes;
    hasNextPage: boolean;
    isFetchingNextPage: boolean;
    fetchNextPage: () => void;
};

// DetailsView isn’t needed, so we’re forcing the cast to prevent TS from complaining about missing properties
export const VIEW_MODE_SETTINGS = {
    [ViewModes.LARGE]: { minItemSize: new Size(300, 300), minSpace: new Size(10, 10), preserveAspectRatio: true },
    [ViewModes.MEDIUM]: { minItemSize: new Size(200, 200), minSpace: new Size(6, 6), preserveAspectRatio: true },
    [ViewModes.SMALL]: { minItemSize: new Size(120, 120), minSpace: new Size(4, 4), preserveAspectRatio: true },
} as Record<ViewModes, GridLayoutOptions>;

export const Gallery = ({ items, viewMode, hasNextPage, isFetchingNextPage, fetchNextPage }: GalleryProps) => {
    const projectId = useProjectIdentifier();
    const { selectedMediaItem, onSelectedMediaItemChange } = useSelectDatasetItem();
    const { selectedKeys, mediaState, setSelectedKeys, toggleSelectedKeys } = useSelectedData();

    const isSetSelectedKeys = selectedKeys instanceof Set;

    return (
        <>
            <VirtualizerGridLayout
                items={items}
                ariaLabel='data-collection-grid'
                selectionMode='multiple'
                mediaState={mediaState}
                selectedKeys={selectedKeys}
                layoutOptions={VIEW_MODE_SETTINGS[viewMode]}
                isLoadingMore={isFetchingNextPage}
                onLoadMore={() => hasNextPage && fetchNextPage()}
                onSelectionChange={setSelectedKeys}
                contentItem={(item) => (
                    <MediaItem
                        contentElement={() => (
                            <MediaThumbnail
                                alt={item.name}
                                url={getThumbnailUrl(projectId, String(item.id))}
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
                            <DeleteMediaItem itemsIds={[String(item.id)]} onDeleted={toggleSelectedKeys} />
                        )}
                    />
                )}
            />

            <DialogContainer type={'fullscreenTakeover'} onDismiss={() => onSelectedMediaItemChange(null)}>
                {selectedMediaItem !== null && (
                    <MediaPreview
                        mediaItem={selectedMediaItem}
                        close={() => onSelectedMediaItemChange(null)}
                        onSelectedMediaItem={onSelectedMediaItemChange}
                    />
                )}
            </DialogContainer>
        </>
    );
};
