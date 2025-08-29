// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { AriaComponentsListBox, DialogContainer, GridLayout, ListBoxItem, Size, View, Virtualizer } from '@geti/ui';

import thumbnailUrl from '../../../assets/mocked-project-thumbnail.png';
import { useSelectedData } from '../../../routes/dataset/provider';
import { CheckboxInput } from '../checkbox-input';
import { InspectDialog } from '../inspect-dialog/inspect-dialog.component';
import { response } from '../mock-response';
import { AnnotationStateIcon } from './annotation-state-icon.component';
import { MediaItem } from './media-item.component';

import classes from './gallery.module.scss';

const layoutOptions = {
    minSpace: new Size(8, 8),
    maxColumns: 8,
    preserveAspectRatio: true,
};

type Item = (typeof response.items)[number];

export const Gallery = () => {
    const [selectedMediaItem, setSelectedMediaItem] = useState<null | Item>(null);
    const { selectedKeys, mediaState, setSelectedKeys } = useSelectedData();
    const isSetSelectedKeys = selectedKeys instanceof Set;

    return (
        <View UNSAFE_className={classes.mainContainer}>
            <Virtualizer layout={GridLayout} layoutOptions={layoutOptions}>
                <AriaComponentsListBox
                    layout='grid'
                    aria-label='data-collection-grid'
                    className={classes.container}
                    selectedKeys={selectedKeys}
                    selectionMode={'multiple'}
                    onSelectionChange={setSelectedKeys}
                >
                    {response.items.map((item) => (
                        <ListBoxItem
                            id={item.id}
                            key={item.id}
                            textValue={item.id}
                            className={classes.mediaItem}
                            data-accepted={mediaState.get(item.id) === 'accepted'}
                            data-rejected={mediaState.get(item.id) === 'rejected'}
                        >
                            <MediaItem
                                contentElement={() => (
                                    <div onDoubleClick={() => setSelectedMediaItem(item)}>
                                        <img
                                            src={thumbnailUrl}
                                            alt={item.original_name}
                                            style={{ objectFit: 'cover', width: '100%', height: '100%' }}
                                        />
                                    </div>
                                )}
                                topRightElement={() => <AnnotationStateIcon state={mediaState.get(item.id)} />}
                                topLeftElement={() => (
                                    <CheckboxInput
                                        isReadOnly
                                        name={`select-${item.id}`}
                                        isChecked={isSetSelectedKeys && selectedKeys.has(item.id)}
                                    />
                                )}
                            />
                        </ListBoxItem>
                    ))}
                </AriaComponentsListBox>
            </Virtualizer>

            <DialogContainer onDismiss={() => setSelectedMediaItem(null)}>
                {selectedMediaItem !== null && (
                    <InspectDialog mediaItem={selectedMediaItem} close={() => setSelectedMediaItem(null)} />
                )}
            </DialogContainer>
        </View>
    );
};
