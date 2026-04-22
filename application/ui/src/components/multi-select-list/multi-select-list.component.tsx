// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentProps, useState } from 'react';

import { Checkbox, Flex, Item, ListView, Selection, Text } from '@geti/ui';

import { isNonEmptyString } from '../../shared/util';

import classes from './multi-select-list.module.scss';

type ListViewProps = ComponentProps<typeof ListView>;

interface MultiSelectListProps
    extends Omit<
        ListViewProps,
        'selectionMode' | 'onSelectionChange' | 'items' | 'defaultSelectedKeys' | 'selectedKeys' | 'children'
    > {
    name: string;
    label?: string;
    defaultSelectedKeys: Set<string>;
    onSelectionChange?: (selectedKeys: Set<string> | 'all') => void;
    items: { id: string; name: string }[];
}

export const MultiSelectList = ({
    name,
    label,
    items,
    onSelectionChange,
    defaultSelectedKeys,
    ...listProps
}: MultiSelectListProps) => {
    const [selectedLabels, setSelectedLabels] = useState<Set<string>>(
        defaultSelectedKeys
        /* () => new Set(allSelectedByDefault ? items.map(({ id }) => id) : []) */
    );

    const allItemSelected = selectedLabels.size === items.length && items.length > 0;

    const handleSelectAllItems = (isSelected: boolean) => {
        const selectedItems = isSelected ? new Set(items.map(({ id }) => id)) : new Set<string>();
        setSelectedLabels(selectedItems);
        onSelectionChange?.(selectedItems);
    };

    const handleSelectChange = (keys: Selection) => {
        const selection = keys === 'all' ? new Set(items.map(({ id }) => id)) : new Set(keys as Set<string>);
        setSelectedLabels(selection);
        onSelectionChange?.(selection);
    };

    return (
        <Flex gap='size-100' direction='column'>
            {isNonEmptyString(label) && <Text UNSAFE_className={classes.label}>{label}</Text>}

            <Checkbox aria-label='Select all items' onChange={handleSelectAllItems} isSelected={allItemSelected}>
                Select all
            </Checkbox>

            <ListView
                {...listProps}
                items={items}
                aria-label={label ?? 'Multi-select list'}
                selectionMode='multiple'
                onSelectionChange={handleSelectChange}
                selectedKeys={selectedLabels}
            >
                {(item) => <Item key={item.id}>{item.name}</Item>}
            </ListView>

            <>
                {Array.from(selectedLabels).map((labelId) => (
                    <input key={labelId} type='hidden' name={name} value={labelId} />
                ))}
            </>
        </Flex>
    );
};
