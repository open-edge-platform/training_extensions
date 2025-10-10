// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentProps, ReactNode, useRef } from 'react';

import { AriaComponentsListBox, GridLayout, ListBoxItem, Loading, Size, View, Virtualizer } from '@geti/ui';
import { useLoadMore } from '@react-aria/utils';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { GridLayoutOptions } from 'react-aria-components';
import { components } from 'src/api/openapi-spec';
import { MediaState, useSelectedData } from 'src/routes/dataset/provider';

import { useGetDatasetItems } from './gallery/use-get-dataset-items.hook';

import classes from './virtualizer-grid-layout.module.scss';

type Item = components['schemas']['DatasetItem'];
type AriaComponentsListBoxProps = ComponentProps<typeof AriaComponentsListBox>;

interface VirtualizerGridLayoutProps extends Pick<AriaComponentsListBoxProps, 'selectedKeys' | 'onSelectionChange'> {
    items: Item[];
    ariaLabel: string;
    mediaState: MediaState;
    selectionMode: 'single' | 'multiple' | 'none';
    layoutOptions: GridLayoutOptions;
    isLoadingMore: boolean;
    onLoadMore: () => void;
    contentItem: (item: Item) => ReactNode;
}

export const VirtualizerGridLayout = ({
    items,
    ariaLabel,
    mediaState,
    selectedKeys,
    isLoadingMore,
    selectionMode,
    layoutOptions,
    onLoadMore,
    contentItem,
    onSelectionChange,
}: VirtualizerGridLayoutProps) => {
    const ref = useRef<HTMLDivElement | null>(null);

    useLoadMore({ isLoading: isLoadingMore, onLoadMore }, ref);

    return (
        <View UNSAFE_className={classes.mainContainer}>
            <Virtualizer layout={GridLayout} layoutOptions={layoutOptions}>
                <AriaComponentsListBox
                    ref={ref}
                    layout='grid'
                    aria-label={ariaLabel}
                    className={classes.container}
                    selectedKeys={selectedKeys}
                    selectionMode={selectionMode}
                    onSelectionChange={onSelectionChange}
                >
                    {items.map((item) => (
                        <ListBoxItem
                            id={item.id}
                            key={`${ariaLabel}-${item.id}`}
                            textValue={item.id}
                            className={classes.mediaItem}
                            data-accepted={mediaState.get(String(item.id)) === 'accepted'}
                            data-rejected={mediaState.get(String(item.id)) === 'rejected'}
                        >
                            {contentItem(item)}
                        </ListBoxItem>
                    ))}
                    {isLoadingMore && (
                        <ListBoxItem id={'loader'} textValue={'loading'}>
                            <Loading mode='overlay' />
                        </ListBoxItem>
                    )}
                </AriaComponentsListBox>
            </Virtualizer>
        </View>
    );
};
