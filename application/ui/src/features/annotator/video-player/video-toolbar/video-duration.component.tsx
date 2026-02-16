// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { formatDurationText } from './time-utils';

export const VideoDuration = () => {
    const currentTime = 0;
    const endTime = 60 * 60 * 1.5;

    const currentFormattedTime = formatDurationText(currentTime);
    const endFormattedTime = formatDurationText(endTime);

    return (
        <span aria-label={'Video duration'}>
            {currentFormattedTime} / {endFormattedTime}
        </span>
    );
};
