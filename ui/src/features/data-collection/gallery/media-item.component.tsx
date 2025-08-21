// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { View } from '@geti/ui';
import { clsx } from 'clsx';
import { isFunction } from 'lodash-es';

import { response } from '../mock-response';
import { getThumbnailUrl } from './utils';

import classes from './media-item.module.scss';

type Item = (typeof response.items)[number];
interface MediaItemProps {
    item: Item;
    topLeftElement?: (item: Item) => ReactNode;
    topRightElement?: (item: Item) => ReactNode;
}

export const MediaItem = ({ item, topLeftElement, topRightElement }: MediaItemProps) => {
    return (
        <View UNSAFE_className={classes.container}>
            <img src={getThumbnailUrl(item.id)} alt={item.original_name} />
            {isFunction(topLeftElement) && (
                <View UNSAFE_className={clsx(classes.leftTopElement, classes.floatingContainer)}>
                    {topLeftElement(item)}
                </View>
            )}
            {isFunction(topRightElement) && (
                <View UNSAFE_className={clsx(classes.rightTopElement, classes.floatingContainer)}>
                    {topRightElement(item)}
                </View>
            )}
        </View>
    );
};
