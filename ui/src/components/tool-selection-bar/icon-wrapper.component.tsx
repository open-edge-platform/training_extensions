// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { clsx } from 'clsx';

import classes from './icon-wrapper.module.scss';

export const IconWrapper = ({
    children,
    onPress,
    isSelected,
}: {
    children: ReactNode;
    onPress?: () => void;
    isSelected?: boolean;
}) => {
    return (
        <div className={clsx(classes.iconWrapper, { [classes.selected]: isSelected })} onClick={onPress}>
            {children}
        </div>
    );
};
