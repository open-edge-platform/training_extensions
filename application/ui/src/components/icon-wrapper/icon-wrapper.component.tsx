// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { clsx } from 'clsx';

import classes from './icon-wrapper.module.scss';

export const IconWrapper = ({
    children,
    onPress,
    isSelected,
    isDisabled,
}: {
    children: ReactNode;
    onPress?: () => void;
    isSelected?: boolean;
    isDisabled?: boolean;
}) => {
    return (
        <div
            className={clsx(classes.iconWrapper, { [classes.selected]: isSelected, [classes.disabled]: isDisabled })}
            onClick={onPress}
        >
            {children}
        </div>
    );
};
