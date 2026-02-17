// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentProps, useState } from 'react';

import { Flex, Item, ListView, Text } from '@geti/ui';

import classes from './multi-select-list.module.scss';

type ListViewProps = ComponentProps<typeof ListView>;

interface MultiSelectListProps
    extends Omit<ListViewProps, 'selectionMode' | 'onSelectionChange' | 'items' | 'defaultSelectedKeys' | 'children'> {
    name: string;
    label: string;
    items: { id: string; name: string }[];
}

export const MultiSelectList = ({ name, label, items, ...listProps }: MultiSelectListProps) => {
    const [selectedLabels, setSelectedLabels] = useState<Set<string>>(new Set());

    return (
        <Flex gap='size-100' direction='column'>
            <Text UNSAFE_className={classes.label}>{label}</Text>

            <ListView
                {...listProps}
                items={items}
                selectionMode='multiple'
                aria-label={label}
                onSelectionChange={(keys) => {
                    const selection = keys === 'all' ? new Set(items.map((l) => l.id)) : new Set(keys as Set<string>);
                    setSelectedLabels(selection);
                }}
            >
                {(item) => <Item>{item.name}</Item>}
            </ListView>

            <>
                {Array.from(selectedLabels).map((labelId) => (
                    <input key={labelId} type='hidden' name={name} value={labelId} />
                ))}
            </>
        </Flex>
    );
};
