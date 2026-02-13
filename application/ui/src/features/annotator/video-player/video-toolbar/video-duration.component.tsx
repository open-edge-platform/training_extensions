// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useDurationText } from './time-utils';

export const VideoDuration = () => {
    const currentTime = 0;
    const endTime = 60 * 60 * 1.5;

    const currentFormattedTime = useDurationText(currentTime);
    const endFormattedTime = useDurationText(endTime);

    return (
        <span aria-label={'Video duration'}>
            {currentFormattedTime} / {endFormattedTime}
        </span>
    );
};
