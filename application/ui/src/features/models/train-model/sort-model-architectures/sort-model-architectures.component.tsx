// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Icon, Item, Picker, Section, Text } from '@geti/ui';
import { SortDown, SortUp } from '@geti/ui/icons';

import { SortingOptions } from './utils';

import styles from './sort-model-architectures.module.scss';

type SortItemType = {
    key: string;
    name: string;
};

type SortWidgetProps = {
    sortBy: SortingOptions;
    onSort: (option: SortingOptions) => void;
    items: SortItemType[][];
    ariaLabel?: string;
};

type SortItemProps = {
    item: {
        key: string;
        name: string;
    };
};

const SortModelArchitectureItem = ({ item }: SortItemProps) => {
    return (
        <>
            <Text>{item.name}</Text>
            <Icon UNSAFE_className={styles.sortModelArchitectureIcon}>
                {item.key.toLocaleLowerCase().endsWith('asc') ? <SortUp /> : <SortDown />}
            </Icon>
        </>
    );
};

export const SortModelArchitectures = ({ sortBy, onSort, items, ariaLabel }: SortWidgetProps) => {
    return (
        <Picker
            isQuiet
            items={items}
            selectedKey={sortBy}
            onSelectionChange={(key) => onSort(key as SortingOptions)}
            aria-label={ariaLabel}
            UNSAFE_className={styles.sortModelArchitectures}
            labelAlign={'start'}
            labelPosition={'side'}
            label={'Sort Models by:'}
        >
            {(item) => {
                return (
                    <Section key={`${item[0].key}-${item[1].key}`}>
                        {item.map((option) => (
                            <Item key={option.key} textValue={option.name}>
                                <SortModelArchitectureItem item={option} />
                            </Item>
                        ))}
                    </Section>
                );
            }}
        </Picker>
    );
};
