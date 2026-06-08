// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import classes from './three-dots-flashing.module.scss';

export const ThreeDotsFlashing = () => {
    return (
        <span aria-hidden='true' className={classes.flashingDots}>
            <span>.</span>
            <span>.</span>
            <span>.</span>
        </span>
    );
};
