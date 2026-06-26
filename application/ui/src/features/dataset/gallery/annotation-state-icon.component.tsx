// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti-ui/ui';
import { CheckCircleOutlined, Search } from '@geti-ui/ui/icons';

import classes from './annotation-state-icon.module.scss';

type AnnotationStatusIconProps = {
    isReviewed: boolean;
};

export const AnnotationStatusIcon = ({ isReviewed }: AnnotationStatusIconProps) => {
    if (isReviewed) {
        return (
            <View UNSAFE_className={classes.reviewed}>
                <CheckCircleOutlined />
            </View>
        );
    }

    return (
        <View UNSAFE_className={classes.toReview}>
            <Search />
        </View>
    );
};
