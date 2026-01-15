// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { DialogContainer, Size } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { CheckboxInput } from '../../../components/checkbox-input/checkbox-input.component';
import type { DatasetItem } from '../../../constants/shared-types';
import { MediaPreview } from '../media-preview/media-preview.component';
import { useSelectedData } from '../selected-data-provider.component';
import { VirtualizerGridLayout } from '../virtualizer-grid-layout/virtualizer-grid-layout.component';
import { AnnotationStatusIcon } from './annotation-state-icon.component';
import { DeleteMediaItem } from './delete-media-item/delete-media-item.component';
import { MediaItem } from './media-item.component';
import { MediaThumbnail } from './media-thumbnail.component';
import { getThumbnailUrl } from './utils';

import classes from './gallery.module.scss';

type GalleryProps = {
    items: DatasetItem[];
    fetchNextPage: () => void;
    hasNextPage: boolean;
    isFetchingNextPage: boolean;
};

const layoutOptions = {
    minSpace: new Size(8, 8),
    maxColumns: 8,
    preserveAspectRatio: true,
};

export const Gallery = ({ items, hasNextPage, isFetchingNextPage, fetchNextPage }: GalleryProps) => {
    const project_id = useProjectIdentifier();

    const [selectedMediaItem, setSelectedMediaItem] = useState<null | DatasetItem>(null);
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
                layoutOptions={layoutOptions}
                isLoadingMore={isFetchingNextPage}
                onLoadMore={() => hasNextPage && fetchNextPage()}
                onSelectionChange={setSelectedKeys}
                contentItem={(item) => (
                    <MediaItem
                        className={classes.mediaItem}
                        contentElement={() => (
                            <MediaThumbnail
                                alt={item.name}
                                url={getThumbnailUrl(project_id, String(item.id))}
                                onDoubleClick={() => setSelectedMediaItem(item)}
                            />
                        )}
                        topLeftElement={() => (
                            <CheckboxInput
                                isReadOnly
                                name={`select-${item.id}`}
                                isChecked={isSetSelectedKeys && selectedKeys.has(String(item.id))}
                            />
                        )}
                        topRightElement={() => (
                            <DeleteMediaItem itemsIds={[String(item.id)]} onDeleted={toggleSelectedKeys} />
                        )}
                        bottomRightElement={() => <AnnotationStatusIcon state={mediaState.get(String(item.id))} />}
                    />
                )}
            />

            <DialogContainer onDismiss={() => setSelectedMediaItem(null)}>
                {selectedMediaItem !== null && (
                    <MediaPreview
                        mediaItem={selectedMediaItem}
                        close={() => setSelectedMediaItem(null)}
                        onSelectedMediaItem={setSelectedMediaItem}
                    />
                )}
            </DialogContainer>
        </>
    );
};
