// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CanceledIcon, CheckCircleOutlined } from '@geti/ui/icons';

import { MediaItemState } from '../../../constants/shared-types';

import classes from './annotation-state-icon.module.scss';

type AnnotationStatusIconProps = {
    state: MediaItemState | undefined;
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
