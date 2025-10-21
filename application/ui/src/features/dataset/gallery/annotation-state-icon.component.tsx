// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CanceledIcon, CheckCircleOutlined } from '@geti/ui/icons';

import { AnnotationStatus } from '../../../routes/dataset/provider';

import classes from './annotation-state-icon.module.scss';

type AnnotationStatusIconProps = {
    state: AnnotationStatus | undefined;
};

export const AnnotationStatusIcon = ({ state }: AnnotationStatusIconProps) => {
    if (state === 'accepted') {
        return <CheckCircleOutlined className={classes.accepted} />;
    }

    if (state === 'rejected') {
        return <CanceledIcon className={classes.rejected} />;
    }

    return null;
};
