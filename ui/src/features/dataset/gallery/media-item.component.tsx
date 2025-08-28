// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { View } from '@geti/ui';
import { clsx } from 'clsx';
import { isFunction } from 'lodash-es';

import classes from './media-item.module.scss';

interface MediaItemProps {
    contentElement: () => ReactNode;
    topLeftElement?: () => ReactNode;
    topRightElement?: () => ReactNode;
}

export const MediaItem = ({ contentElement, topLeftElement, topRightElement }: MediaItemProps) => {
    return (
        <View width={'100%'}>
            {contentElement()}

            {isFunction(topLeftElement) && (
                <View UNSAFE_className={clsx(classes.leftTopElement, classes.floatingContainer)}>
                    {topLeftElement()}
                </View>
            )}

            {isFunction(topRightElement) && (
                <View UNSAFE_className={clsx(classes.rightTopElement, classes.floatingContainer)}>
                    {topRightElement()}
                </View>
            )}
        </View>
    );
};
