// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useRef, useState } from 'react';

import {
    AriaComponentsListBox,
    DialogContainer,
    GridLayout,
    ListBoxItem,
    Loading,
    Size,
    View,
    Virtualizer,
} from '@geti/ui';
import { useLoadMore } from '@react-aria/utils';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { AnnotationActionsProvider } from 'src/features/annotator/annotation-actions-provider.component';
import { AnnotatorProvider } from 'src/features/annotator/annotator-provider.component';

import { CheckboxInput } from '../../../components/checkbox-input/checkbox-input.component';
import { useSelectedData } from '../../../routes/dataset/provider';
import { DatasetItem } from '../../annotator/types';
import { MediaPreview } from '../media-preview/media-preview.component';
import { AnnotationStateIcon } from './annotation-state-icon.component';
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
    const ref = useRef<HTMLDivElement | null>(null);
    const project_id = useProjectIdentifier();

    const [selectedMediaItem, setSelectedMediaItem] = useState<null | DatasetItem>(null);
    const { selectedKeys, mediaState, setSelectedKeys, toggleSelectedKeys } = useSelectedData();

    const isSetSelectedKeys = selectedKeys instanceof Set;

    useLoadMore(
        {
            isLoading: isFetchingNextPage,
            onLoadMore: () => hasNextPage && fetchNextPage(),
        },
        ref
    );

    return (
        <View UNSAFE_className={classes.mainContainer}>
            <Virtualizer layout={GridLayout} layoutOptions={layoutOptions}>
                <AriaComponentsListBox
                    ref={ref}
                    layout='grid'
                    aria-label='data-collection-grid'
                    className={classes.container}
                    selectedKeys={selectedKeys}
                    selectionMode={'multiple'}
                    onSelectionChange={setSelectedKeys}
                >
                    {items.map((item) => (
                        <ListBoxItem
                            id={item.id}
                            key={item.id}
                            textValue={item.id}
                            className={classes.mediaItem}
                            data-accepted={mediaState.get(String(item.id)) === 'accepted'}
                            data-rejected={mediaState.get(String(item.id)) === 'rejected'}
                        >
                            <MediaItem
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
                                bottomRightElement={() => (
                                    <AnnotationStateIcon state={mediaState.get(String(item.id))} />
                                )}
                            />
                        </ListBoxItem>
                    ))}
                    {isFetchingNextPage && (
                        <ListBoxItem id={'loader'} textValue={'loading'}>
                            <Loading mode='overlay' />
                        </ListBoxItem>
                    )}
                </AriaComponentsListBox>
            </Virtualizer>

            <DialogContainer onDismiss={() => setSelectedMediaItem(null)}>
                {selectedMediaItem !== null && (
                    <Suspense fallback={<Loading size='L' />}>
                        <AnnotatorProvider mediaItem={selectedMediaItem}>
                            <AnnotationActionsProvider>
                                <MediaPreview mediaItem={selectedMediaItem} close={() => setSelectedMediaItem(null)} />
                            </AnnotationActionsProvider>
                        </AnnotatorProvider>
                    </Suspense>
                )}
            </DialogContainer>
        </View>
    );
};
