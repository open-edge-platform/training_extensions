// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CanceledIcon, CheckCircleOutlined } from '@geti/ui/icons';

import { AnnotationState } from '../../../routes/data-collection/provider';

import classes from './annotation-state-icon.module.scss';

type AnnotationStateIconProps = {
    state: AnnotationState | undefined;
};

export const AnnotationStateIcon = ({ state }: AnnotationStateIconProps) => {
    if (state === 'accepted') {
        return <CheckCircleOutlined className={classes.accepted} />;
    }

    if (state === 'rejected') {
        return <CanceledIcon className={classes.rejected} />;
    }

    return null;
};
