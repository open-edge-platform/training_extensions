// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';
import { CanceledIcon, CheckCircleOutlined, Search } from '@geti/ui/icons';

import type { MediaItemState } from '../../../constants/shared-types';

import classes from './annotation-state-icon.module.scss';

type AnnotationStatusIconProps = {
    state: MediaItemState | undefined;
};

export const AnnotationStatusIcon = ({ state }: AnnotationStatusIconProps) => {
    if (state === 'reviewed') {
        return (
            <View UNSAFE_className={classes.reviewed}>
                <CheckCircleOutlined />
            </View>
        );
    }

    if (state === 'rejected') {
        return (
            <View UNSAFE_className={classes.rejected}>
                <CanceledIcon />
            </View>
        );
    }

    return (
        <View UNSAFE_className={classes.toReview}>
            <Search />
        </View>
    );
};
