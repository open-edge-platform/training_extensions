// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentProps, ReactNode, useRef } from 'react';

import { AriaComponentsListBox, GridLayout, ListBoxItem, Loading, View, Virtualizer } from '@geti/ui';
import { useLoadMore } from '@react-aria/utils';
import { GridLayoutOptions } from 'react-aria-components';

import { useGetTargetPosition } from './use-get-target-position.hook';

import classes from './virtualizer-grid-layout.module.scss';

type AriaComponentsListBoxProps = ComponentProps<typeof AriaComponentsListBox>;

interface GridItem {
    id: string;
    [key: string]: unknown;
}

interface VirtualizerGridLayoutProps<T extends GridItem>
    extends Pick<AriaComponentsListBoxProps, 'selectedKeys' | 'onSelectionChange'> {
    items: T[];
    ariaLabel: string;
    scrollToIndex?: number;
    selectionMode: 'single' | 'multiple' | 'none';
    layoutOptions: GridLayoutOptions;
    isPending?: boolean;
    isLoadingMore: boolean;
    onLoadMore: () => void;
    contentItem: (item: T) => ReactNode;
    getItemId?: (item: T) => string | number;
}

const MIN_SPACE = 18; // default value for GridLayoutOptions.minSpace.height

export const VirtualizerGridLayout = <T extends GridItem>({
    items,
    ariaLabel,
    selectedKeys,
    isPending = false,
    isLoadingMore,
    selectionMode,
    layoutOptions,
    scrollToIndex,
    onLoadMore,
    contentItem,
    onSelectionChange,
    getItemId = (item) => item.id,
}: VirtualizerGridLayoutProps<T>) => {
    const ref = useRef<HTMLDivElement | null>(null);

    useLoadMore({ isLoading: isLoadingMore, onLoadMore }, ref);

    useGetTargetPosition({
        ref,
        delay: 40,
        gap: layoutOptions.minSpace?.height ?? MIN_SPACE,
        scrollToIndex,
        callback: (top) => {
            ref.current?.scrollTo({ top, behavior: 'smooth' });
        },
    });

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
                    {items.map((item, index) => {
                        const itemId = getItemId(item);

                        return (
                            <ListBoxItem
                                id={itemId}
                                key={`${ariaLabel}-${itemId}-${index}`}
                                textValue={String(itemId)}
                                className={classes.mediaItem}
                            >
                                {contentItem(item)}
                            </ListBoxItem>
                        );
                    })}
                    {isLoadingMore && (
                        <ListBoxItem id={'loader'} textValue={'loading'}>
                            <Loading mode='overlay' />
                        </ListBoxItem>
                    )}
                </AriaComponentsListBox>
            </Virtualizer>
            {isPending && <Loading mode='overlay' />}
        </View>
    );
};
